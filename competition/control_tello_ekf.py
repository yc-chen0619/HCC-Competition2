#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String
from tello_msgs.srv import TelloAction
import math
import yaml
import os
import random
import pygame
from ament_index_python.packages import get_package_share_directory

class ControlTelloPygame(Node):
    def __init__(self):
        super().__init__('control_tello_pygame')
        
        # --- ROS 2 通訊端設定 ---
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.client_ = self.create_client(TelloAction, '/tello_action')
        self.ekf_sub = self.create_subscription(PoseWithCovarianceStamped, '/ekf_pose', self.ekf_callback, 10)
        self.target_sub = self.create_subscription(PoseStamped, '/balloon/target_pose', self.target_callback, 10)

        # --- 讀取 AprilTag 地圖資料 ---
        package_share = get_package_share_directory('tello_localization') # 請確保名稱與你的地圖 package 一致
        yaml_path = os.path.join(package_share, 'map', 'apriltag_map.yaml')
        try:
            with open(yaml_path, 'r') as f:
                self.tag_map = yaml.safe_load(f)['tags']
            self.get_logger().info("✅ 成功載入 AprilTag 地圖！")
        except Exception as e:
            self.get_logger().error(f"❌ 無法載入 AprilTag 地圖: {e}")
            self.tag_map = []

        # --- 初始化 Pygame 視窗 ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 500
        self.screen_height = 400
        self.screen = pygame.display.set_set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Tello 競賽控制儀表板")
        self.font = pygame.font.SysFont("Courier", 20)
        self.font_bold = pygame.font.SysFont("Courier", 24, bold=True)

        # --- 狀態機與控制變數 ---
        self.state = 'IDLE' 
        self.detected_tag_id = "None"
        
        # 狀態機內部計時器
        self.takeoff_counter = 0  

        # 座標狀態
        self.drone_x = self.drone_y = self.drone_z = self.drone_yaw = 0.0
        self.target_x = self.target_y = self.target_z = 0.0
        self.last_target_time = None
        self.target_timeout = 0.5 

        # 導航目標點
        self.goal_x = self.goal_y = self.goal_z = self.goal_yaw = 0.0

        # PID 與安全限速參數
        self.KP_XY = 0.6        
        self.KP_Z = 0.8         
        self.KP_YAW = 1.0       
        self.MAX_VEL = 0.15      # 安全限速 (m/s)
        self.MAX_YAW_RATE = 0.25 # 最大旋轉速度 (rad/s)
        self.SEARCH_YAW_RATE = 0.3 # 轉圈找球速度
        self.BRAKE_DISTANCE = 0.45 # 離氣球多近要煞車 (公尺)

        # 即時輸出的控制指令
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.yaw_rate = 0.0

        # 以 20Hz (每 0.05 秒) 驅動核心控制、鍵盤讀取與畫面渲染
        self.timer = self.create_timer(0.05, self.timer_callback)

    def ekf_callback(self, msg):
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = math.atan2(siny_cosp, cosy_cosp)

    def target_callback(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        self.last_target_time = self.get_clock().now()

    def send_tello_cmd(self, cmd_string):
        if not self.client_.service_is_ready():
            self.get_logger().warn(f"Tello Service 未就緒: {cmd_string}")
            return
        request = TelloAction.Request()
        request.cmd = cmd_string
        self.client_.call_async(request)
        self.get_logger().info(f"發送指令: {cmd_string}")

    def calculate_tag_goal(self, tag_id):
        for tag in self.tag_map:
            if tag['id'] == tag_id:
                t_x = tag['position'][0]
                t_y = tag['position'][1]
                t_z = tag['position'][2]
                t_yaw = tag['orientation_rpy'][2] 
                
                # 前方延伸 30cm ~ 50cm (此處設為 0.4m 作為折衷值)
                self.goal_x = t_x + 0.4 * math.cos(t_yaw)
                self.goal_y = t_y + 0.4 * math.sin(t_yaw)
                self.goal_z = t_z  
                self.goal_yaw = t_yaw + math.pi # 面對 Tag
                return True
        return False

    def handle_keyboard(self):
        """ 讀取 Pygame 鍵盤事件 """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.emergency_shutdown()

            elif event.type == pygame.KEYDOWN:
                # 1. 空白鍵：瞬間速度歸零，切回 IDLE 停懸
                if event.key == pygame.K_SPACE:
                    self.state = 'IDLE'
                    self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0
                    self.get_logger().info("🛑 [緊急煞車] 速度歸零，進入原地停懸。")

                # 2. 按 S：起飛並進入搜尋
                elif event.key == pygame.K_s:
                    if self.state == 'IDLE':
                        self.state = 'TAKEOFF'
                        self.takeoff_counter = 0
                        self.send_tello_cmd('takeoff')
                        self.get_logger().info("🚀 [熱鍵 S] 觸發起飛流程...")

                # 3. 按 G：前往衝撞氣球
                elif event.key == pygame.K_g:
                    self.state = 'TRACK_BALLOON'
                    self.get_logger().info("🎯 [熱鍵 G] 啟動氣球追蹤衝撞！")

                # 4. 按 C：分類 (隨機產生 Tag ID) 並停懸
                elif event.key == pygame.K_c:
                    self.state = 'CLASSIFY'
                    # 隨機從你們地圖有的 Tag 裡挑一個 (假設 ID 是 1, 2, 3)
                    available_tags = [tag['id'] for tag in self.tag_map] if self.tag_map else [1, 2, 3]
                    self.detected_tag_id = random.choice(available_tags)
                    self.get_logger().info(f"🔮 [熱鍵 C] 分類完成！識別為停機坪 Tag ID: {self.detected_tag_id}")
                    # 分類完畢，自動切回 IDLE 停懸等待下一步指令
                    self.state = 'IDLE'

                # 5. 按 V：精準導航到分類的 Tag 前方降落
                elif event.key == pygame.K_v:
                    if self.detected_tag_id == "None":
                        self.get_logger().error("❌ 無法執行導航：尚未進行分類 (請先按 C)！")
                    else:
                        if self.calculate_tag_goal(int(self.detected_tag_id)):
                            self.state = 'NAVIGATE_TO_TAG'
                            self.get_logger().info(f"🛫 [熱鍵 V] 開始導航至 Tag {self.detected_tag_id} 前方。")
                        else:
                            self.get_logger().error(f"❌ 地圖中找不到 Tag {self.detected_tag_id}")

    def timer_callback(self):
        # 處理鍵盤與視窗退出事件
        self.handle_keyboard()

        # 檢查氣球觀測是否有效
        is_target_valid = False
        if self.last_target_time is not None:
            dt = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
            if dt <= self.target_timeout:
                is_target_valid = True

        # ====================================================
        # 自動化狀態機核心控制邏輯
        # ====================================================
        if self.state == 'IDLE':
            self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0

        elif self.state == 'TAKEOFF':
            self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0
            self.takeoff_counter += 1
            if self.takeoff_counter >= 80: # 4 秒穩定時間
                self.state = 'SEARCH_BALLOON'

        elif self.state == 'SEARCH_BALLOON':
            # 沒看到球就原地自轉
            self.v_x = self.v_y = self.v_z = 0.0
            if not is_target_valid:
                self.yaw_rate = self.SEARCH_YAW_RATE

        elif self.state == 'TRACK_BALLOON':
            if not is_target_valid:
                self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0  # 搞丟目標時先停懸安全
            else:
                dx_map = self.target_x - self.drone_x
                dy_map = self.target_y - self.drone_y
                dz_map = self.target_z - self.drone_z
                distance = math.sqrt(dx_map**2 + dy_map**2 + dz_map**2)

                # 4. 條件限制：如果距離小於設定的安全煞車距離，強制煞車停懸
                if distance < self.BRAKE_DISTANCE:
                    self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0
                else:
                    # PID 計算並轉換至機身座標系
                    self.v_x = dx_map * math.cos(self.drone_yaw) + dy_map * math.sin(self.drone_yaw)
                    self.v_y = -dx_map * math.sin(self.drone_yaw) + dy_map * math.cos(self.drone_yaw)
                    self.v_z = dz_map

                    self.v_x *= self.KP_XY
                    self.v_y *= self.KP_XY
                    self.v_z *= self.KP_Z

                    target_yaw = math.atan2(dy_map, dx_map)
                    dyaw = target_yaw - self.drone_yaw
                    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                    self.yaw_rate = self.KP_YAW * dyaw

                    # 安全速限保護
                    self.v_x = max(min(self.v_x, self.MAX_VEL), -self.MAX_VEL)
                    self.v_y = max(min(self.v_y, self.MAX_VEL), -self.MAX_VEL)
                    self.v_z = max(min(self.v_z, self.MAX_VEL), -self.MAX_VEL)
                    self.yaw_rate = max(min(self.yaw_rate, self.MAX_YAW_RATE), -self.MAX_YAW_RATE)

        elif self.state == 'NAVIGATE_TO_TAG':
            dx_map = self.goal_x - self.drone_x
            dy_map = self.goal_y - self.drone_y
            dz_map = self.goal_z - self.drone_z
            
            # 轉換至機身座標系
            self.v_x = dx_map * math.cos(self.drone_yaw) + dy_map * math.sin(self.drone_yaw)
            self.v_y = -dx_map * math.sin(self.drone_yaw) + dy_map * math.cos(self.drone_yaw)
            self.v_z = dz_map
            
            self.v_x *= self.KP_XY
            self.v_y *= self.KP_XY
            self.v_z *= self.KP_Z
            
            dyaw = self.goal_yaw - self.drone_yaw
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            self.yaw_rate = self.KP_YAW * dyaw
            
            # 安全速限
            self.v_x = max(min(self.v_x, self.MAX_VEL), -self.MAX_VEL)
            self.v_y = max(min(self.v_y, self.MAX_VEL), -self.MAX_VEL)
            self.v_z = max(min(self.v_z, self.MAX_VEL), -self.MAX_VEL)
            self.yaw_rate = max(min(self.yaw_rate, self.MAX_YAW_RATE), -self.MAX_YAW_RATE)

            # 6. 抵達判斷（在 x, y, z 誤差範圍內做自動 land）
            if (dx_map**2 + dy_map**2 < 0.12**2) and (abs(dz_map) < 0.12) and (abs(dyaw) < 0.2):
                self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0
                self.send_tello_cmd('land')
                self.state = 'IDLE'
                self.get_logger().info("🎉 精準抵達停機坪點！執行自動降落。")

        # --- 發布 Tello 速度指令 ---
        twist = Twist()
        twist.linear.x = float(self.v_x)
        twist.linear.y = float(self.v_y)
        twist.linear.z = float(self.v_z)
        twist.angular.z = float(self.yaw_rate)
        self.publisher_.publish(twist)

        # --- 渲染更新 Pygame 視窗圖形 ---
        self.draw_dashboard(is_target_valid)

    def draw_dashboard(self, is_target_valid):
        """ 在視窗上刷出即時狀態數據與警告 """
        self.screen.fill((25, 25, 25)) # 深灰背景

        # 1. 顯示目前狀態模式
        state_color = (0, 255, 0) if self.state == 'IDLE' else (255, 165, 0)
        state_text = self.font_bold.render(f"STATE: {self.state}", True, state_color)
        self.screen.blit(state_text, (20, 20))

        # 2. 顯示即時控制速度 (v_x, v_y, v_z, yaw_rate)
        vel_text_1 = self.font.render(f"cmd_vx : {self.v_x:.3f} m/s", True, (200, 200, 200))
        vel_text_2 = self.font.render(f"cmd_vy : {self.v_y:.3f} m/s", True, (200, 200, 200))
        vel_text_3 = self.font.render(f"cmd_vz : {self.v_z:.3f} m/s", True, (200, 200, 200))
        vel_text_4 = self.font.render(f"cmd_yaw: {self.yaw_rate:.3f} rad/s", True, (200, 200, 200))
        self.screen.blit(vel_text_1, (20, 60))
        self.screen.blit(vel_text_2, (20, 85))
        self.screen.blit(vel_text_3, (20, 110))
        self.screen.blit(vel_text_4, (20, 135))

        # 3. 顯示目前無人機的世界座標偏角 (EKF)
        pose_text = self.font.render(f"Drone XYZ: [{self.drone_x:.2f}, {self.drone_y:.2f}, {self.drone_z:.2f}]", True, (100, 200, 255))
        yaw_text  = self.font.render(f"Drone Yaw: {math.degrees(self.drone_yaw):.1f}°", True, (100, 200, 255))
        self.screen.blit(pose_text, (20, 175))
        self.screen.blit(yaw_text, (20, 200))

        # 4. 顯示辨識分類結果 Tag ID
        tag_text = self.font_bold.render(f"Target Tag ID: {self.detected_tag_id}", True, (255, 255, 0))
        self.screen.blit(tag_text, (20, 240))

        # 5. 顯示警告提示：有沒有抓到紅色氣球
        if is_target_valid:
            balloon_msg = self.font_bold.render("!!! 搜尋到紅色氣球 !!!", True, (255, 50, 50))
            self.screen.blit(balloon_msg, (20, 290))
        else:
            balloon_msg = self.font.render("Scanning for balloon...", True, (120, 120, 120))
            self.screen.blit(balloon_msg, (20, 290))

        # 6. 底部操作指引快捷鍵
        help_text = self.font.render("[SPACE]Hover [S]Takeoff [G]Track [C]Classify [V]Nav", True, (150, 150, 150))
        self.screen.blit(help_text, (10, 360))

        pygame.display.flip()

    def emergency_shutdown(self):
        """ 處理關閉視窗或例外時的緊急降落 """
        self.get_logger().warn("🚨 偵測到退出事件！發送緊急 Land 指令...")
        self.state = 'IDLE'
        emergency_twist = Twist()
        self.publisher_.publish(emergency_twist)
        self.send_tello_cmd('land')
        pygame.quit()
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = ControlTelloPygame()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        # 1. 滿足需求：無論如何遇到 Ctrl+C，強制發布 land
        print("\n🛑 [警告] 偵測到終端機 Ctrl+C 中斷！強制發送降落指令。")
    finally:
        # 強制雙重保險降落
        emergency_twist = Twist()
        node.publisher_.publish(emergency_twist)
        node.send_tello_cmd('land')
        
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()