import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

try:
    from ros2_aruco_interfaces.msg import ArucoMarkers
    _HAS_ARUCO_PKG = True
except ImportError:
    _HAS_ARUCO_PKG = False

from planning.ik import IKPlanner

PREGRASP_Z_OFFSET = 0.16
GRASP_Z_OFFSET    = 0.095
DISCOVERY_POS     = (0.6, 0.0, 0.1)             # CHANGE IF NEEDED
DROP_Z            = -0.40
ARUCO_TOPIC       = '/aruco_markers'
ARUCO_TARGET_ID   = 10                          # CHANGE IF NEEDED
COLOR_TOPIC       = '/scout/package_request'
DONE_TOPIC        = '/scout/package_ready'

BLOCK_TOPICS = {
    'red':    '/red_block_pose',
    'blue':   '/blue_block_pose',
    'yellow': '/yellow_block_pose',
}

class UR7e_ColoredCubeGrasp(Node):
    def __init__(self):
        super().__init__('colored_cube_grasp')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(String, COLOR_TOPIC, self._color_cmd_cb, 1)

        self.block_poses = {'red': None, 'blue': None, 'yellow': None}
        for color, topic in BLOCK_TOPICS.items():
            self.create_subscription(
                PointStamped, topic,
                lambda msg, c=color: self._block_pose_cb(c, msg),
                1,
            )

        self.create_subscription(JointState, '/joint_states', self._js_cb, 1)

        if _HAS_ARUCO_PKG:
            self.create_subscription(ArucoMarkers, ARUCO_TOPIC, self._aruco_cb, 10)
        else:
            self.get_logger().warn('ros2_aruco_interfaces not found — ArUco detection disabled.')

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        self.done_pub = self.create_publisher(Bool, DONE_TOPIC, 1)
        self.ik_planner  = IKPlanner()

        self.joint_state = None
        self.home_joint_state = None
        self.busy = False
        self.discovering = False
        self.aruco_pos = None
        self.job_queue = []

        self.get_logger().info(f'UR7e_ColoredCubeGrasp ready. Waiting for color command on {COLOR_TOPIC}.')

    def _js_cb(self, msg: JointState):
        self.joint_state = msg
        if self.home_joint_state is None:
            self.home_joint_state = msg

    def _block_pose_cb(self, color: str, msg: PointStamped):
        if not self.busy:
            self.block_poses[color] = (msg.point.x, msg.point.y, msg.point.z)

    def _color_cmd_cb(self, msg: String):
        color = msg.data.strip().lower()
        if color not in ('red', 'blue', 'yellow'):
            self.get_logger().warn(f'Unknown color command: "{msg.data}" — ignoring.')
            return
        if self.busy:
            self.get_logger().warn(f'Received "{color}" but arm is busy — ignoring.')
            return
        self.get_logger().info(f'Received color command: {color}')
        self._start_task(color)

    def _aruco_cb(self, msg):
        if not self.discovering:
            return

        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            if int(marker_id) != ARUCO_TARGET_ID:
                continue

            self.get_logger().info(
                f'ArUco {marker_id} RAW ({msg.header.frame_id}): '
                f'x={pose.position.x:.3f} y={pose.position.y:.3f} z={pose.position.z:.3f}'
            )

            ps = PoseStamped()
            ps.header.frame_id = 'camera_depth_optical_frame'
            ps.header.stamp    = rclpy.time.Time().to_msg()
            ps.pose = pose
            try:
                transformed = self.tf_buffer.transform(
                    ps, 'base_link',
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
                tx, ty, tz = (transformed.pose.position.x,
                              transformed.pose.position.y,
                              transformed.pose.position.z)
                self.aruco_pos = (tx, ty, tz)
                self.get_logger().info(
                    f'ArUco {marker_id} -> base_link: x={tx:.3f} y={ty:.3f} z={tz:.3f}'
                )
            except Exception as exc:
                self.get_logger().warn(f'TF transform failed for marker {marker_id}: {exc}')
                return

            self._queue_drop_jobs()
            break

    def _abort(self):
        self.busy = False
        self.discovering = False
        self.job_queue.clear()

    def _start_task(self, color: str):
        if self.busy:
            return

        if self.joint_state is None:
            self.get_logger().warn('No joint state yet — retrying in 1 s.')
            def _retry_js():
                self.destroy_timer(timer_js)
                self._start_task(color)
            timer_js = self.create_timer(1.0, _retry_js)
            return

        block_pos = self.block_poses[color]
        if block_pos is None:
            self.get_logger().warn(f'No {color} block detected yet — retrying in 1 s.')
            def _retry_bp():
                self.destroy_timer(timer_bp)
                self._start_task(color)
            timer_bp = self.create_timer(1.0, _retry_bp)
            return

        self.busy = True
        self.discovering = False
        self.aruco_pos   = None
        self.job_queue.clear()

        bx, by, bz = block_pos
        self.get_logger().info(f'Starting task: pick {color} block at ({bx:.3f}, {by:.3f}, {bz:.3f})')

        seed = self.joint_state

        def _ik(x, y, z, label):
            nonlocal seed
            js = self.ik_planner.compute_ik(seed, x, y, z)
            if js is None:
                self.get_logger().error(f'IK failed for {label} ({x:.3f},{y:.3f},{z:.3f})')
            else:
                seed = js
            return js

        dx, dy, dz = DISCOVERY_POS

        pre_grasp = _ik(bx, by, bz + PREGRASP_Z_OFFSET, 'pre-grasp')
        if pre_grasp:
            self.job_queue.append(pre_grasp)

        grasp = _ik(bx - 0.01, by, bz + GRASP_Z_OFFSET, 'grasp')
        if grasp:
            self.job_queue.append(grasp)

        self.job_queue.append('close_gripper')

        lift = _ik(bx, by, bz + PREGRASP_Z_OFFSET, 'lift')
        if lift:
            self.job_queue.append(lift)

        discovery = _ik(dx, dy, dz, 'discovery')
        if discovery:
            self.job_queue.append(discovery)

        self.job_queue.append('discover_aruco')

        self.execute_jobs()

    def _queue_drop_jobs(self):
        self.discovering = False

        ax, ay, az = self.aruco_pos
        dist = (ax**2 + ay**2 + DROP_Z**2) ** 0.5
        self.get_logger().info(
            f'ArUco drop target (base_link): x={ax:.3f} y={ay:.3f} z={DROP_Z:.3f} '
            f'(reach dist={dist:.3f} m)'
        )

        seed = self.joint_state

        def _ik(x, y, z, label):
            nonlocal seed
            js = self.ik_planner.compute_ik(seed, x, y, z)
            if js is None:
                self.get_logger().error(f'IK failed for {label} ({x:.3f},{y:.3f},{z:.3f})')
            else:
                seed = js
            return js

        drop = _ik(ax - 0.065, ay + 0.04, DROP_Z, 'drop')
        if drop:
            self.job_queue.append(drop)
            self.job_queue.append('open_gripper')
        else:
            self.get_logger().error(
                f'DROP POSITION UNREACHABLE: ({ax:.3f}, {ay:.3f}, {DROP_Z:.3f}) — '
                'skipping drop. Verify ArUco is in base_link frame and within reach.'
            )

        dx, dy, dz = DISCOVERY_POS
        retreat = _ik(dx, dy, dz, 'retreat')
        if retreat:
            self.job_queue.append(retreat)

        if self.home_joint_state is not None:
            self.job_queue.append(self.home_joint_state)

        self.job_queue.append('publish_done')
        self.execute_jobs()

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):

            traj = self.ik_planner.plan_to_joints(next_job, self.joint_state)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                self._abort()
                return
            
            self.get_logger().info("Planned to position")
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == 'open_gripper':
            self.get_logger().info("Opening gripper")
            self._toggle_gripper()

        elif next_job == 'close_gripper':
            self.get_logger().info("Closing gripper")
            self._toggle_gripper()

        elif next_job == 'discover_aruco':
            self.get_logger().info(f'Arm at DISCOVERY — waiting for ArUco tag #{ARUCO_TARGET_ID}...')
            self.discovering = True

        elif next_job == 'publish_done':
            done_msg = Bool()
            done_msg.data = True
            self.done_pub.publish(done_msg)
            self.get_logger().info(f'Task complete. Published True on {DONE_TOPIC}.')
            self.busy = False

        else:
            self.get_logger().error(f'Unknown job type: {next_job!r}')
            self.execute_jobs()

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available.')
            self._abort()
            return
        
        future = self.gripper_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()

    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected.')
            self._abort()
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            self._abort()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_ColoredCubeGrasp()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
