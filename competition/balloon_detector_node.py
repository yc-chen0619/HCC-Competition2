#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge

# 引入 TF2 相關工具，用於做座標系轉換
import tf2_ros
import tf2_geometry_msgs 

def CameraIntrinsics():
    FX = 907.45
    FY = 906.73
    CX = 470.05
    CY = 369.95
    TagSize = 0.20                   # m
    return FX, FY, CX, CY, TagSize

class BalloonDetectorNode(Node):
    def __init__(self):
        super().__init__('balloon_detector_node')
        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.balloon_raw_pub = self.create_publisher(PoseStamped, '/balloon/pose_raw', 10)
        
        # 賽前參數設定
        camera_intrinsic = CameraIntrinsics()
        self.FX = camera_intrinsic[0]
        self.FY = camera_intrinsic[1]
        self.CX = camera_intrinsic[2]
        self.CY = camera_intrinsic[3]
        self.F_AVG = (self.FX + self.FY) / 2.0 # 平均焦距
        
        self.declare_parameter('balloon_diameter', 0.25)
        self.BALLOON_D = self.get_parameter('balloon_diameter').value

        # 建立 TF2 監聽器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("Balloon Detector Node (面積等效法) 已啟動！")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge 轉換失敗: {e}")
            return

        # 1. HSV 紅色遮罩處理 (因紅色在兩端，需合併兩個範圍)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # 進行形態學去雜訊
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 2. 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return # 沒看到紅球則跳過

        # 找到面積最大的輪廓
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area < 150: # 面積太小視為雜訊
            return

        # 3. 輪廓面積等效法計算相對 3D 座標
        # 像素半徑 r = sqrt(Area / pi) -> 像素直徑 d = 2 * r
        pixel_diameter = 2.0 * math.sqrt(area / math.pi)
        
        # 計算深度 Z_c = (f * D) / d
        Z_c = (self.F_AVG * self.BALLOON_D) / pixel_diameter

        # 計算中心點像素座標 (u, v)
        M = cv2.moments(max_contour)
        if M['m00'] == 0: return
        u = M['m10'] / M['m00']
        v = M['m01'] / M['m00']

        # 計算相機座標系下的 X_c, Y_c
        X_c = ((u - self.CX) / self.FX) * Z_c
        Y_c = ((v - self.CY) / self.FY) * Z_c

        # 4. 使用 TF2 將相機系下的點轉到 map 系下
        point_cam = PointStamped()
        point_cam.header.frame_id = 'camera_frame' # 沿用你 launch 檔定義的 camera_frame
        point_cam.header.stamp = msg.header.stamp  # 用影像的時間戳同步
        point_cam.point.x = float(X_c)
        point_cam.point.y = float(Y_c)
        point_cam.point.z = float(Z_c)

        try:
            # 查找最新 map 到 camera_frame 的轉換關係
            transform = self.tf_buffer.lookup_transform('map', 'camera_frame', rclpy.time.Time())
            point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)
            
            # 發布未濾波的全局氣球 Pose
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'map'
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.pose.position = point_map.point
            # 球體無方向性，四元數給預設
            pose_msg.pose.orientation.w = 1.0
            
            self.balloon_raw_pub.publish(pose_msg)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warn(f"TF 座標轉換失敗: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = BalloonDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()