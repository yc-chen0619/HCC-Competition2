#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

class BalloonTrackingNode(Node):
    def __init__(self):
        super().__init__('balloon_tracking_node')
        self.raw_sub = self.create_subscription(PoseStamped, '/balloon/pose_raw', self.balloon_cb, 10)
        self.target_pub = self.create_publisher(PoseStamped, '/balloon/target_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/balloon/markers', 10)
        
        # predict time horizon
        self.declare_parameter('lookahead_time', 0.8) # 可根據 Tello 的反應速度動態調整
        self.dt_pred = self.get_parameter('lookahead_time').value

        # balloon state = [x, y, z, vx, vy, vz]^T
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0  # 初始不確定度大一點
        
        # 過程雜訊 Q (假設速度容易受到氣流干擾，雜訊給稍微大一點)
        self.Q = np.eye(6) * 0.02
        self.Q[3:, 3:] *= 2.0 
        # 觀測雜訊 R (影像測距可能會有小幅抖動)
        self.R = np.eye(3) * 0.05

        self.is_initialized = False
        self.last_time = None

        # Preventing Ghosting
        self.last_measurement_time = self.get_clock().now()
        self.timeout_duration = 1.5                                        # Time horizon of disappear (s)
        self.check_timer = self.create_timer(0.1, self.check_timeout_loop) # 10Hz for check

        self.get_logger().info("Balloon Tracking Node (KF 速度預測) 已啟動！")

    def check_timeout_loop(self):
        if not self.is_initialized:
            return
            
        current_time = self.get_clock().now()
        dt = (current_time - self.last_measurement_time).nanoseconds / 1e9
        if dt > self.timeout_duration:
            self.get_logger().warn(f"氣球丟失！超過 {self.timeout_duration} 秒未偵測到，停止預測與追蹤...")
            self.is_initialized = False # 重置 KF 狀態
            self.clear_rviz_markers()

    def balloon_cb(self, msg):
        self.last_measurement_time = self.get_clock().now()
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # initial first balloon positiion
        if not self.is_initialized:
            self.x[0, 0] = msg.pose.position.x
            self.x[1, 0] = msg.pose.position.y
            self.x[2, 0] = msg.pose.position.z
            self.x[3:, 0] = 0.0                  # initial velocity setting -> 0
            self.last_time = current_time
            self.is_initialized = True
            return

        dt = current_time - self.last_time
        if dt <= 0: return
        self.last_time = current_time

        # z = [x, y, z]^T
        z = np.array([[msg.pose.position.x],
                      [msg.pose.position.y],
                      [msg.pose.position.z]])

        # Predict (等速模型)
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Update
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # calculate t + dt_pred timestep position
        pred_x = self.x[0, 0] + self.x[3, 0] * self.dt_pred
        pred_y = self.x[1, 0] + self.x[4, 0] * self.dt_pred
        pred_z = self.x[2, 0] + self.x[5, 0] * self.dt_pred

        # publish predict balloon position
        target_msg = PoseStamped()
        target_msg.header = msg.header
        target_msg.pose.position.x = float(pred_x)
        target_msg.pose.position.y = float(pred_y)
        target_msg.pose.position.z = float(pred_z)
        target_msg.pose.orientation.w = 1.0
        self.target_pub.publish(target_msg)

        # visualization
        self.publish_rviz_markers(msg.header, pred_x, pred_y, pred_z)

    def clear_rviz_markers(self):
        marker_array = MarkerArray()
        m_delete = Marker()
        m_delete.action = Marker.DELETEALL
        marker_array.markers.append(m_delete)
        self.marker_pub.publish(marker_array)

    def publish_rviz_markers(self, header, px, py, pz):
        marker_array = MarkerArray()

        # Marker 1: 當前被濾波平滑後的氣球位置 (綠色球體)
        m_balloon = Marker()
        m_balloon.header = header
        m_balloon.id = 100
        m_balloon.type = Marker.SPHERE
        m_balloon.action = Marker.ADD
        m_balloon.pose.position.x = float(self.x[0, 0])
        m_balloon.pose.position.y = float(self.x[1, 0])
        m_balloon.pose.position.z = float(self.x[2, 0])
        m_balloon.scale.x = 0.25
        m_balloon.scale.y = 0.25
        m_balloon.scale.z = 0.25
        m_balloon.color.r = 0.0
        m_balloon.color.g = 1.0 # 綠色代表追蹤中
        m_balloon.color.b = 0.0
        m_balloon.color.a = 0.8
        marker_array.markers.append(m_balloon)

        # Marker 2: 未來預測位置 (紅色球體，代表 Tello 的狙擊點)
        m_target = Marker()
        m_target.header = header
        m_target.id = 101
        m_target.type = Marker.SPHERE
        m_target.action = Marker.ADD
        m_target.pose.position.x = float(px)
        m_target.pose.position.y = float(py)
        m_target.pose.position.z = float(pz)
        m_target.scale.x = 0.2
        m_target.scale.y = 0.2
        m_target.scale.z = 0.2
        m_target.color.r = 1.0 # 紅色代表攔截點
        m_target.color.g = 0.0
        m_target.color.b = 0.0
        m_target.color.a = 1.0
        marker_array.markers.append(m_target)

        # Marker 3: 速度向量箭頭 (從氣球指向預測位置)
        m_arrow = Marker()
        m_arrow.header = header
        m_arrow.id = 102
        m_arrow.type = Marker.ARROW
        m_arrow.action = Marker.ADD
        m_arrow.points = [m_balloon.pose.position, m_target.pose.position]
        m_arrow.scale.x = 0.03 # 軸直徑
        m_arrow.scale.y = 0.06 # 箭頭直徑
        m_arrow.scale.z = 0.1  # 箭頭長度
        m_arrow.color.r = 1.0
        m_arrow.color.g = 1.0
        m_arrow.color.b = 0.0
        m_arrow.color.a = 1.0
        marker_array.markers.append(m_arrow)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = BalloonTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()