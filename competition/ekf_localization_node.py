import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from tello_msgs.msg import FlightData
from nav_msgs.msg import Path

import tf2_ros
import numpy as np
import math

from scipy.spatial.transform import Rotation as R


class EKFLocalizationNode(Node):
    def __init__(self):
        super().__init__('ekf_localization_node')
        self.subscription = self.create_subscription(PoseArray, '/apriltag/detections', self.detection_callback, 10)
        #self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.flight_sub = self.create_subscription(FlightData, '/flight_data', self.flight_data_callback, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.path_pub = self.create_publisher(Path, '/ekf_path', 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        self.mu = np.zeros((6,1))
        self.Sigma = np.eye(6) * 0.1
        self.u = np.zeros((6,1))
        # [x, y, z, roll, yaw, pitch]
        self.Rm = np.diag([0.001, 0.001, 0.005, 0.01, 0.01, 0.01])
        # [x, y, z, roll, yaw, pitch]
        self.Q = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])

        self.dt = 0.1
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.predict_timer)

        # intial tello control
        self.last_flight_time = None
        self.last_roll = 0.0
        self.last_pitch = 0.0
        self.last_yaw = 0.0

    # state : [x, y, z, roll, yaw, pitch]
    # control : [v_x, v_y, v_z, roll_rate, yaw_rate, pitch_rate]
    def motion_model(self, x, u):
        dt = self.dt

        px = x[0,0]
        py = x[1,0]
        pz = x[2,0]
        roll = x[3,0]
        yaw = x[4,0]
        pitch = x[5,0]

        vx = u[0,0]
        vy = u[1,0]
        vz = u[2,0]
        roll_rate = u[3,0]
        yaw_rate = u[4,0]
        pitch_rate = u[5,0]

        x_pred = np.zeros((6,1))
        x_pred[0,0] = px + (vx * np.cos(yaw) - vy * np.sin(yaw)) * dt
        x_pred[1,0] = py + (vx * np.sin(yaw) + vy * np.cos(yaw)) * dt
        x_pred[2,0] = pz + vz * dt
        x_pred[3,0] = roll + roll_rate * dt
        x_pred[4,0] = yaw + yaw_rate * dt
        x_pred[5,0] = pitch + pitch_rate * dt
        return x_pred

    def jacobian_F(self, x, u):
        dt = self.dt
        yaw = x[4,0]
        vx = u[0,0]
        vy = u[1,0]

        F = np.eye(6)
        F[0,4] = (-vx * np.sin(yaw) - vy * np.cos(yaw)) * dt
        F[1,4] = ( vx * np.cos(yaw) - vy * np.sin(yaw)) * dt
        return F
    
    def predict(self, u):
        F = self.jacobian_F(self.mu, u)
        self.mu = self.motion_model(self.mu, u)
        self.Sigma = F @ self.Sigma @ F.T + self.Rm

    def update(self, z):
        H = np.eye(6)
        z_pred = self.mu
        y = z - z_pred
        y[3,0] = (y[3,0] + np.pi) % (2 * np.pi) - np.pi
        y[4,0] = (y[4,0] + np.pi) % (2 * np.pi) - np.pi
        y[5,0] = (y[5,0] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.Sigma @ H.T + self.Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        I = np.eye(6)
        self.Sigma = (I - K @ H) @ self.Sigma

    def predict_timer(self):
        now = self.get_clock().now()
        dt_duration = now - self.last_time
        
        self.dt = dt_duration.nanoseconds / 1e9 
        self.last_time = now

        if self.dt > 0.0 and self.dt < 1.0:
            self.predict(self.u)
        self.publish_pose()

    def cmd_callback(self, msg):
        self.u[0, 0] = msg.linear.x    # vx
        self.u[1, 0] = msg.linear.y    # vy
        self.u[2, 0] = msg.linear.z    # vz
        self.u[3, 0] = msg.angular.x   # roll_rate
        self.u[4, 0] = msg.angular.z   # yaw_rate
        self.u[5, 0] = msg.angular.y   # pitch_rate
    
    def flight_data_callback(self, msg):
        # (cm/s) -> (m/s)、FRD frame -> FLU frame
        vx = msg.vgx / 100.0
        vy = -msg.vgy / 100.0  # Tello 右為正 -> ROS 左為正
        vz = -msg.vgz / 100.0  # Tello 下為正 -> ROS 上為正
        # Degree -> Radian
        now = self.get_clock().now()
        roll_rad = math.radians(msg.roll)
        pitch_rad = math.radians(msg.pitch)
        yaw_rad = math.radians(msg.yaw)

        self.u[0, 0] = vx
        self.u[1, 0] = vy
        self.u[2, 0] = vz
        if self.last_flight_time is not None:
            dt = (now - self.last_flight_time).nanoseconds / 1e9
            if dt > 0.001:
                d_roll = (roll_rad - self.last_roll + math.pi) % (2 * math.pi) - math.pi
                d_pitch = (pitch_rad - self.last_pitch + math.pi) % (2 * math.pi) - math.pi
                d_yaw = (yaw_rad - self.last_yaw + math.pi) % (2 * math.pi) - math.pi

                # w = dA / dt
                self.u[3, 0] = d_roll / dt
                self.u[4, 0] = d_yaw / dt
                self.u[5, 0] = d_pitch / dt

        self.last_flight_time = now
        self.last_roll = roll_rad
        self.last_pitch = pitch_rad
        self.last_yaw = yaw_rad

    def detection_callback(self, msg):
        if len(msg.poses) == 0:
            return

        pose = msg.poses[0]
        quat = [pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w]
        
        r = R.from_quat(quat)
        euler = r.as_euler('xyz') # roll, pitch, yaw
        z = np.array([
            [pose.position.x],
            [pose.position.y],
            [pose.position.z],
            [euler[0]],
            [euler[2]],
            [euler[1]]
        ])
        self.update(z)

    def publish_pose(self):
        now = self.get_clock().now().to_msg()

        # 建立帶有共變異數的 PoseWithCovarianceStamped 訊息
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = now
        msg.pose.pose.position.x = float(self.mu[0,0])
        msg.pose.pose.position.y = float(self.mu[1,0])
        msg.pose.pose.position.z = float(self.mu[2,0])
        # 將尤拉角轉回四元數發布
        q = R.from_euler('xyz', [self.mu[3,0], self.mu[5,0], self.mu[4,0]]).as_quat().tolist()
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        # 對調成 ROS 標準的 [x(0), y(1), z(2), roll(3), pitch(4), yaw(5)] 順序
        idx_mapping = [0, 1, 2, 3, 5, 4]
        ros_sigma = self.Sigma[np.ix_(idx_mapping, idx_mapping)]
        msg.pose.covariance = ros_sigma.flatten().tolist()
        self.pose_pub.publish(msg)
        
        # Publish Path
        path_pose = PoseStamped()
        path_pose.header.frame_id = 'map'
        path_pose.header.stamp = now
        path_pose.pose.position.x = float(self.mu[0,0])
        path_pose.pose.position.y = float(self.mu[1,0])
        path_pose.pose.position.z = float(self.mu[2,0])
        path_pose.pose.orientation.x = q[0]
        path_pose.pose.orientation.y = q[1]
        path_pose.pose.orientation.z = q[2]
        path_pose.pose.orientation.w = q[3]
        self.path_msg.poses.append(path_pose)
        self.path_msg.header.stamp = now
        self.path_pub.publish(self.path_msg)

        # Broadcast TF (map -> base_link)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.mu[0,0])
        t.transform.translation.y = float(self.mu[1,0])
        t.transform.translation.z = float(self.mu[2,0])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()