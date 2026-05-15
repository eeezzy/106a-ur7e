import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from rcl_interfaces.msg import SetParametersResult

CLUSTER_RADIUS  = 0.08
MIN_CLUSTER_PTS = 40


def _greedy_cluster(points):
    unassigned = np.ones(len(points), dtype=bool)
    clusters = []
    while unassigned.sum() >= MIN_CLUSTER_PTS:
        seed_idx = int(np.where(unassigned)[0][0])
        remaining = np.where(unassigned)[0]
        dists = np.linalg.norm(points[remaining, :2] - points[seed_idx, :2], axis=1)
        members = remaining[dists <= CLUSTER_RADIUS]
        unassigned[members] = False
        if len(members) >= MIN_CLUSTER_PTS:
            clusters.append(members)
    return clusters


def _classify_color(r, g, b):
    if r >= g and r >= b:
        return 'red'
    elif b >= r and b >= g:
        return 'blue'
    else:
        return 'yellow'


class ColoredBlockPCSubscriber(Node):
    def __init__(self):
        super().__init__('colored_block_perception')

        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.max_y = float(self.declare_parameter('max_y',  0.72).value)
        self.min_z = float(self.declare_parameter('min_z', -0.14).value)
        self.max_z = float(self.declare_parameter('max_z', -0.11).value)
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PointCloud2, 
            '/camera/camera/depth/color/points', 
            self._pc_cb, 
            10
        )

        self._pubs = {
            'red':    self.create_publisher(PointStamped, '/red_block_pose',    1),
            'blue':   self.create_publisher(PointStamped, '/blue_block_pose',   1),
            'yellow': self.create_publisher(PointStamped, '/yellow_block_pose', 1),
        }
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info("Subscribed to PointCloud2 topic and marker publisher ready")

    def _pc_cb(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, 'camera_depth_optical_frame', Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform camera_depth_optical_frame to {self.target_frame}: {ex}')
            return

        transformed_cloud = do_transform_cloud(msg, tf)

        raw_points = pc2.read_points(
            transformed_cloud, 
            field_names=('x', 'y', 'z'), 
            skip_nans=True
        )

        xyz = np.column_stack(
                (raw_points['x'], raw_points['y'], raw_points['z'])
            ).astype(np.float32, copy=False)

        raw_rgb = pc2.read_points(transformed_cloud, field_names=('rgb',), skip_nans=True)
        rgb_int = raw_rgb['rgb'].copy().view(np.uint32)
        r_all = ((rgb_int >> 16) & 0xFF).astype(np.float32)
        g_all = ((rgb_int >>  8) & 0xFF).astype(np.float32)
        b_all = (rgb_int & 0xFF).astype(np.float32)

        mask = (xyz[:, 2] >= self.min_z) & (xyz[:, 2] <= self.max_z) & (xyz[:, 1] <= self.max_y)
        xyz, r_all, g_all, b_all = xyz[mask], r_all[mask], g_all[mask], b_all[mask]

        if xyz.size == 0:
            self.get_logger().debug('No points after filter.')
            return

        self.filtered_points_pub.publish(pc2.create_cloud_xyz32(transformed_cloud.header, xyz.tolist()))

        clusters = _greedy_cluster(xyz)
        if not clusters:
            return

        now = self.get_clock().now().to_msg()
        for idx in clusters:
            r_m, g_m, b_m = r_all[idx].mean(), g_all[idx].mean(), b_all[idx].mean()
            color = _classify_color(r_m, g_m, b_m)
            centroid = xyz[idx].mean(axis=0)
            pt = PointStamped()
            pt.header.frame_id = self.target_frame
            pt.header.stamp    = now
            pt.point.x, pt.point.y, pt.point.z = map(float, centroid)
            self._pubs[color].publish(pt)

    def _on_parameter_update(self, params):
        new_min_z, new_max_z, new_max_y = self.min_z, self.max_z, self.max_y
        
        for param in params:
            if param.name == 'min_z' and param.type_ == Parameter.Type.DOUBLE:
                new_min_z = float(param.value)
            elif param.name == 'max_z' and param.type_ == Parameter.Type.DOUBLE:
                new_max_z = float(param.value)
            elif param.name == 'max_y' and param.type_ == Parameter.Type.DOUBLE:
                new_max_y = float(param.value)

        if new_min_z > new_max_z:
            return SetParametersResult(
                successful=False, 
                reason='min_z must be <= max_z',
            )
        
        self.min_z, self.max_z, self.max_y = new_min_z, new_max_z, new_max_y
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = ColoredBlockPCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
