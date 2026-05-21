import rclpy
from rclpy.node import Node
import os
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from cv_bridge import CvBridge

import cv2
import numpy as np
import yaml

from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R


def CameraIntrinsics():
    FX = 907.45
    FY = 906.73
    CX = 470.05
    CY = 369.95
    TagSize = 0.20 # m
    return FX, FY, CX, CY, TagSize


class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('apriltag_detector_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pose_pub = self.create_publisher(PoseArray, '/apriltag/detections', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/apriltag/markers', 10)
        self.detector = Detector(families='tag36h11', 
                                 nthreads=1, 
                                 quad_decimate=1.0, 
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)
        self.get_logger().info("Image subscriber with pupil_apriltags initialized.")

        camera_intrinsic = CameraIntrinsics()
        self.fx = camera_intrinsic[0]
        self.fy = camera_intrinsic[1]
        self.cx = camera_intrinsic[2]
        self.cy = camera_intrinsic[3]
        self.tag_size = camera_intrinsic[4]
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]

        package_share = get_package_share_directory('tello_localization')
        yaml_path = os.path.join(package_share, 'map', 'apriltag_map.yaml')
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)['tags']
                self.tag_pose_dict = {tag['id']: tag for tag in yaml_data}
        except Exception as e:
            self.get_logger().error(f"Failed to load yaml: {e}")
            self.tag_pose_dict = {}

    def get_tag_world_pose(self, tag_id):
        tag = self.tag_pose_dict.get(tag_id)
        if tag is None:
            self.get_logger().warn(f"No world pose defined for tag ID {tag_id}")
            return None

        pos = tag['position']
        rpy = tag['orientation_rpy']
        rot = R.from_euler('xyz', rpy).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
    
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray,
                                    estimate_tag_pose=True,
                                    camera_params=self.camera_params,
                                    tag_size=self.tag_size)

        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for idx, tag in enumerate(tags):
            # T_c_t (T_camera_tag)
            T_c_t = np.eye(4)
            T_c_t[:3, :3] = tag.pose_R
            T_c_t[:3, 3] = tag.pose_t.flatten()

            # T_w_t (T_world_tag)
            T_w_t = self.get_tag_world_pose(tag.tag_id)
            if T_w_t is None:
                continue 

            # OpenCV-coordinates transformed to ROS-coordinates (X-forward, Y-left, Z-up)
            # T_w_c (T_world_camera)
            T_w_c_opencv = T_w_t @ np.linalg.inv(T_c_t)
            R_cv_to_ros = np.array([
                [0, -1,  0,  0],
                [0,  0, -1,  0],
                [1,  0,  0,  0],
                [0,  0,  0,  1]
            ])
            T_w_c = T_w_c_opencv @ R_cv_to_ros

            # Camera in World
            pose = Pose()
            pose.position.x = float(T_w_c[0, 3])
            pose.position.y = float(T_w_c[1, 3])
            pose.position.z = float(T_w_c[2, 3])
            q = R.from_matrix(T_w_c[:3, :3]).as_quat()
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            pose_array.poses.append(pose)


            # AprilTag in World
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = int(tag.tag_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(T_w_t[0, 3])
            marker.pose.position.y = float(T_w_t[1, 3])
            marker.pose.position.z = float(T_w_t[2, 3]) + 0.001
            q_tag = R.from_matrix(T_w_t[:3, :3]).as_quat()
            marker.pose.orientation.x = q_tag[0]
            marker.pose.orientation.y = q_tag[1]
            marker.pose.orientation.z = q_tag[2]
            marker.pose.orientation.w = q_tag[3]
            marker.scale.x = self.tag_size
            marker.scale.y = self.tag_size
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        if len(pose_array.poses) > 0:
            self.pose_pub.publish(pose_array)
            self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
