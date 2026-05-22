#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from tello_msgs.srv import TelloAction
import pygame
import sys
import math

class ControlTelloEKF(Node):
    def __init__(self):
        super().__init__('control_tello_ekf')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.client_ = self.create_client(TelloAction, '/tello_action')
        self.ekf_sub = self.create_subscription(PoseWithCovarianceStamped, '/ekf_pose', self.ekf_callback, 10)
        self.target_sub = self.create_subscription(PoseStamped, '/balloon/target_pose', self.target_callback, 10)
        
        # 儲存座標狀態
        self.drone_x = None
        self.drone_y = None
        self.drone_z = None
        self.drone_yaw = None
        self.target_x = None
        self.target_y = None
        self.target_z = None

        #超過 0.5 秒沒收到 target_pose 就強制懸停
        self.last_target_time = None
        self.target_timeout = 0.5   #秒

        # 控制模式：False 為手動鍵盤，True 為自動追蹤氣球
        self.auto_mode = False

        # 自動控制增益 (P 控制器參數，可根據實驗微調穩定度)
        self.KP_XY = 0.6    # 水平移動增益
        self.KP_Z = 0.8     # 高度控制增益
        self.KP_YAW = 1.0   # 轉向對準氣球增益
        self.MAX_VEL = 0.5  # 限制自動模式最大速度 (公尺/秒)，安全第一
        self.MAX_YAW_RATE = 0.8 # 限制最大旋轉角速度 (rad/s)，避免自旋失控

        pygame.init()
        self.screen = pygame.display.set_mode((480, 460))
        pygame.display.set_caption('Tello & EKF Controller (GTA 5 Mode + Auto)')
        self.font = pygame.font.SysFont(None, 24)
        
        # initial control signals (6 DoF)
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.roll_rate  = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate   = 0.0

        self.speed_step = 0.3
        self.angle_step = 0.5

        # scanning in 20Hz
        self.timer = self.create_timer(0.05, self.timer_callback)

    def ekf_callback(self, msg):
        # 讀取無人機當前世界座標位置
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        
        # 從四元數轉換出目前的 Yaw 角
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = math.atan2(siny_cosp, cosy_cosp)

    def target_callback(self, msg):
        # 讀取氣球預測的未來攔截世界座標
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        self.last_target_time = self.get_clock().now()

    def send_tello_cmd(self, cmd_string):
        if not self.client_.service_is_ready():
            self.get_logger().warn(f"Tello Action Service 未就緒，無法發送: {cmd_string}")
            return
        request = TelloAction.Request()
        request.cmd = cmd_string
        self.client_.call_async(request)
        self.get_logger().info(f"已發送 Service 指令: {cmd_string}")

    def timer_callback(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.KEYDOWN:
                # tello action mod change (takeoff & land)
                if event.key == pygame.K_t: 
                    self.send_tello_cmd('takeoff')
                elif event.key == pygame.K_l: 
                    self.send_tello_cmd('land')

                # 新增：模式切換按鍵 [M]
                elif event.key == pygame.K_m:
                    self.auto_mode = not self.auto_mode
                    if self.auto_mode:
                        self.get_logger().info("切換至：自動氣球追蹤模式！")
                    else:
                        self.get_logger().info("切換至：手動鍵盤控制模式。")
                        self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0

                # Stop function keys / 安全覆寫開關
                if event.key == pygame.K_SPACE: 
                    self.auto_mode = False # 強制關閉自動模式
                    self.v_x = self.v_y = self.v_z = 0.0
                    self.roll_rate = self.pitch_rate = self.yaw_rate = 0.0

                # 以下鍵盤手動控制，僅在手動模式下觸發
                if not self.auto_mode:
                    # Left hand (WASD)
                    if event.key == pygame.K_w: self.v_z = self.speed_step
                    elif event.key == pygame.K_s: self.v_z = -self.speed_step
                    elif event.key == pygame.K_a: self.yaw_rate = self.angle_step
                    elif event.key == pygame.K_d: self.yaw_rate = -self.angle_step
                    
                    # Right hand (8456)
                    elif event.key in [pygame.K_KP8, pygame.K_8]: 
                        self.v_x = self.speed_step
                        self.pitch_rate = self.angle_step
                    elif event.key in [pygame.K_KP5, pygame.K_5]: 
                        self.v_x = -self.speed_step
                        self.pitch_rate = -self.angle_step
                    elif event.key in [pygame.K_KP4, pygame.K_4]: 
                        self.v_y = self.speed_step
                        self.roll_rate = self.angle_step
                    elif event.key in [pygame.K_KP6, pygame.K_6]: 
                        self.v_y = -self.speed_step
                        self.roll_rate = -self.angle_step
                        
            elif event.type == pygame.KEYUP and not self.auto_mode:
                if event.key in [pygame.K_w, pygame.K_s]: self.v_z = 0.0
                elif event.key in [pygame.K_a, pygame.K_d]: self.yaw_rate = 0.0
                elif event.key in [pygame.K_KP8, pygame.K_8, pygame.K_KP5, pygame.K_5]: 
                    self.v_x = 0.0
                    self.pitch_rate = 0.0
                elif event.key in [pygame.K_KP4, pygame.K_4, pygame.K_KP6, pygame.K_6]: 
                    self.v_y = 0.0
                    self.roll_rate = 0.0

        # ====================================================
        # 自動模式核心控制邏輯 (核心運算)
        # ====================================================
        if self.auto_mode:
            # 檢查 target 是否過期 (搭配 tracking_node 的丟失不發布機制)
            is_target_valid = False
            if self.last_target_time is not None:
                dt = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
                if dt <= self.target_timeout:
                    is_target_valid = True

            # 確保有無人機定位，且氣球目標有效且沒過期
            if (self.drone_x is not None) and is_target_valid:
                # 1. 計算在世界座標系(Map Frame)下的誤差
                dx_map = self.target_x - self.drone_x
                dy_map = self.target_y - self.drone_y
                dz_map = self.target_z - self.drone_z

                # 2. 關鍵幾何轉換：將 Map 誤差向量旋轉至無人機的機身座標系 (Body Frame)
                # Tello cmd_vel 吃的是機身座標 (vx向前, vy向左)
                self.v_x = dx_map * math.cos(self.drone_yaw) + dy_map * math.sin(self.drone_yaw)
                self.v_y = -dx_map * math.sin(self.drone_yaw) + dy_map * math.cos(self.drone_yaw)
                self.v_z = dz_map

                # 3. 乘上 P 增益
                self.v_x *= self.KP_XY
                self.v_y *= self.KP_XY
                self.v_z *= self.KP_Z

                # 4. 轉向控制：計算讓相機主動朝向氣球所需的 Yaw 角度誤差
                target_yaw = math.atan2(dy_map, dx_map)
                dyaw = target_yaw - self.drone_yaw
                # 將角度誤差標準化至 [-pi, pi] 區間，防止盲目亂轉旋轉
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                raw_yaw_rate = self.KP_YAW * dyaw
                self.yaw_rate = max(min(raw_yaw_rate, self.MAX_YAW_RATE), -self.MAX_YAW_RATE)

                # 5. 安全限速保護
                self.v_x = max(min(self.v_x, self.MAX_VEL), -self.MAX_VEL)
                self.v_y = max(min(self.v_y, self.MAX_VEL), -self.MAX_VEL)
                self.v_z = max(min(self.v_z, self.MAX_VEL), -self.MAX_VEL)
                
                # 自動模式下保持姿態穩定率為 0
                self.roll_rate = 0.0
                self.pitch_rate = 0.0
            else:
                # 資料未齊全時懸停
                self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0

        # --- 更新 Pygame 視窗顯示面板 ---
        self.screen.fill((40, 44, 52)) # 深色背景
        mode_str = "AUTO TRACKING" if self.auto_mode else "MANUAL KEYBOARD"
        mode_color = (152, 195, 121) if self.auto_mode else (229, 192, 123) # 綠色 vs 黃色

        info_text = [
            " [ Tello EKF Controller : GTA 5 Mode ]",
            " * Keep this window focused to control *",
            " ---------------------------------------",
            f"  MODE STATUS : {mode_str}",
            "  [M] : TOGGLE MANUAL / AUTO TRACKING ",
            "  [T] : TAKEOFF  /  [L] : LAND ",
            " ---------------------------------------",
            " --- Current Body Outputs ---",
            f"   v_x (Forward) : {self.v_x:.2f}",
            f"   v_y (Leftward) : {self.v_y:.2f}",
            f"   v_z (Upward)  : {self.v_z:.2f}",
            f"   yaw_rate      : {self.yaw_rate:.2f}",
            "",
            " Press SPACE to EMERGENCY STOP / MANUAL MODE"
        ]
        
        y_offset = 15
        for line in info_text:
            color = mode_color if "MODE STATUS" in line else (220, 220, 220)
            text_surface = self.font.render(line, True, color)
            self.screen.blit(text_surface, (20, y_offset))
            y_offset += 28
        pygame.display.flip()

        velocity_print = f"\r[{mode_str}] cmd = [vx:{self.v_x:.2f}, vy:{self.v_y:.2f}, vz:{self.v_z:.2f}, "
        ryp_rate_print = f"yaw_rate:{self.yaw_rate:.2f}]"
        sys.stdout.write(velocity_print + ryp_rate_print)
        sys.stdout.flush()

        # publish control to Tello
        twist = Twist()
        twist.linear.x = float(self.v_x)
        twist.linear.y = float(self.v_y)
        twist.linear.z = float(self.v_z)
        twist.angular.x = float(self.roll_rate)
        twist.angular.y = float(self.pitch_rate)
        twist.angular.z = float(self.yaw_rate)
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ControlTelloEKF()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.publisher_.publish(twist)
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()