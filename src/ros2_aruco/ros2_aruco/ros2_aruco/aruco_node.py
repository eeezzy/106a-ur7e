"""
Detects ArUco markers (DICT_5X5_250, 5 cm) and publishes their ids and poses.

Subscriptions:
   /camera/image_raw      (sensor_msgs/Image)
   /camera/camera_info    (sensor_msgs/CameraInfo)

Published Topics:
   aruco_poses   (geometry_msgs/PoseArray)   — all detected marker poses
   aruco_markers (ros2_aruco_interfaces/ArucoMarkers) — poses + ids

Published Transforms:
   camera_frame -> ar_marker_{id}  for each detected marker

Author: Nathan Sprague
Version: 10/26/2020
Edited by: I Gy Chen
"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tf2_ros import TransformBroadcaster


MARKER_SIZE_M = 0.05


def quaternion_from_matrix(matrix):
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )
        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )
        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )
        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        dictionary_id_name = self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        image_topic        = self.get_parameter("image_topic").get_parameter_value().string_value
        info_topic         = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.camera_frame  = self.get_parameter("camera_frame").get_parameter_value().string_value

        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(f"bad aruco_dictionary_id: {dictionary_id_name}")
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error(f"valid options: {options}")

        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.poses_pub   = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.info_msg      = None
        self.intrinsic_mat = None
        self.distortion    = None

        self.aruco_dictionary  = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters  = cv2.aruco.DetectorParameters_create()
        self.bridge            = CvBridge()

        self.get_logger().info(
            f"ArucoNode ready — dictionary={dictionary_id_name}, "
            f"marker_size={MARKER_SIZE_M} m, image_topic={image_topic}"
        )

    def info_callback(self, info_msg):
        self.info_msg      = info_msg
        self.intrinsic_mat = np.reshape(np.array(info_msg.k), (3, 3))
        self.distortion    = np.array(info_msg.d)
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info received yet.")
            return

        frame_id = self.camera_frame if self.camera_frame else self.info_msg.header.frame_id

        cv_image   = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        corners, marker_ids, _ = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )

        markers    = ArucoMarkers()
        pose_array = PoseArray()
        markers.header.frame_id    = frame_id
        pose_array.header.frame_id = frame_id
        markers.header.stamp       = img_msg.header.stamp
        pose_array.header.stamp    = img_msg.header.stamp

        if marker_ids is None:
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE_M, self.intrinsic_mat, self.distortion
        )

        for i, marker_id in enumerate(marker_ids):
            pose = Pose()
            pose.position.x = tvecs[i][0][0]
            pose.position.y = tvecs[i][0][1]
            pose.position.z = tvecs[i][0][2]

            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            quat = quaternion_from_matrix(rot_matrix)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            transform = TransformStamped()
            transform.header.stamp    = img_msg.header.stamp
            transform.header.frame_id = frame_id
            transform.child_frame_id  = f"ar_marker_{marker_id[0]}"
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]
            self.tf_broadcaster.sendTransform(transform)

            pose_array.poses.append(pose)
            markers.poses.append(pose)
            markers.marker_ids.append(marker_id[0])

        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(markers)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
