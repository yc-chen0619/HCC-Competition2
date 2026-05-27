#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String
from tello_msgs.srv import TelloAction
import math
import yaml
import os
from ament_index_python.packages import get_package_share_directory

class ControlTelloEKF(Node):
    def __init__(self):
        super().__init__('control_tello_ekf')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.client_ = self.create_client(TelloAction, '/tello_action')
        self.ekf_sub = self.create_subscription(PoseWithCovarianceStamped, '/ekf_pose', self.ekf_callback, 10)
        self.target_sub = self.create_subscription(PoseStamped, '/balloon/target_pose', self.target_callback, 10)
        self.cmd_sub = self.create_subscription(String, '/competition/cmd', self.command_callback, 10)

        # --- 狀態機核心變數 ---
        # 預設為 IDLE (原地懸停待命)，安全第一
        self.state = 'IDLE' 
        self.takeoff_counter = 0  # 計算起飛等待時間
        
        # --- 讀取 AprilTag 地圖資料 (用於 Step 2 導航) ---
        package_share = get_package_share_directory('competition')
        yaml_path = os.path.join(package_share, 'map', 'apriltag_map.yaml')
        try:
            with open(yaml_path, 'r') as f:
                self.tag_map = yaml.safe_load(f)['tags']
            self.get_logger().info("✅ 成功載入 AprilTag 地圖 YAML 檔案！")
        except Exception as e:
            self.get_logger().error(f"❌ 無法載入 AprilTag 地圖: {e}")
            self.tag_map = []

        # --- 座標與時間狀態 ---
        self.drone_x = self.drone_y = self.drone_z = self.drone_yaw = None
        # balloon position in global
        self.target_x = self.target_y = self.target_z = None
        self.last_target_time = None
        self.target_timeout = 0.5 # (s)
        # landing position in global
        self.goal_x = self.goal_y = self.goal_z = self.goal_yaw = None
        
        # --- PID 控制器與安全參數 (可根據實測微調) ---
        self.KP_XY = 0.6        # 水平移動增益
        self.KP_Z = 0.8         # 高度控制增益
        self.KP_YAW = 1.0       # 轉向對準增益
        self.MAX_VEL = 0.1      # 限制自動模式最大速度 (m/s)
        self.MAX_YAW_RATE = 0.2 # 限制最大旋轉角速度 (rad/s)
        self.SEARCH_YAW_RATE = 0.3 # 自動找球時的原地自轉速度

        # 控制指令變數
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.yaw_rate = 0.0

        # 以 20Hz (每 0.05 秒) 執行核心控制迴圈
        self.timer = self.create_timer(0.05, self.timer_callback)

    def ekf_callback(self, msg):
        """ 讀取無人機當前世界座標與 Yaw 角 """
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = math.atan2(siny_cosp, cosy_cosp)

    def target_callback(self, msg):
        """ 讀取氣球預測的未來攔截世界座標 """
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        self.last_target_time = self.get_clock().now()

    def command_callback(self, msg):
        """ 接收終端機指令以切換階段 """
        cmd = msg.data.strip().lower()
        
        if cmd == 'step1':
            self.state = 'TAKEOFF'
            self.takeoff_counter = 0
            self.send_tello_cmd('takeoff')
            self.get_logger().info("🚀 [指令] 收到 step1！無人機開始起飛...")
            
        elif cmd.startswith('step2_'):
            # 例如收到 'step2_1' 代表辨識結果為 1 號停機坪
            try:
                tag_id = int(cmd.split('_')[1])
                if self.calculate_tag_goal(tag_id):
                    self.state = 'STEP2_NAVIGATE'
                    self.get_logger().info(f"🚀 [指令] 開始執行第二階段：前往停機坪 Tag {tag_id}！")
            except Exception as e:
                self.get_logger().error(f"指令解析失敗: {e}")
                
        elif cmd == 'idle':
            self.state = 'IDLE'
            self.get_logger().info("🛑 [指令] 強制切換回 IDLE 原地停懸待命。")
        
        elif cmd == 'land':
            self.send_tello_cmd('land')
            self.get_logger().info("🚀 [指令] 收到降落...")

    def send_tello_cmd(self, cmd_string):
        """ 發送 Tello 控制動作（如 land） """
        if not self.client_.service_is_ready():
            self.get_logger().warn(f"Tello Action Service 未就緒，無法發送: {cmd_string}")
            return
        request = TelloAction.Request()
        request.cmd = cmd_string
        self.client_.call_async(request)
        self.get_logger().info(f"已發送 Tello 指令: {cmd_string}")

    def calculate_tag_goal(self, tag_id):
        """ 計算 Tag 前方 30cm 的導航目標點 """
        for tag in self.tag_map:
            if tag['id'] == tag_id:
                t_x = tag['position'][0]
                t_y = tag['position'][1]
                t_z = tag['position'][2]
                t_yaw = tag['orientation_rpy'][2] 
                
                # 幾何計算：向 Tag 的前方延伸 30 公分 (0.3m)
                # 註：請根據你們場地實際設定微調加減號
                self.goal_x = t_x + 0.3 * math.cos(t_yaw)
                self.goal_y = t_y + 0.3 * math.sin(t_yaw)
                self.goal_z = t_z  
                self.goal_yaw = t_yaw + math.pi # 機頭正面對準 Tag
                return True
        self.get_logger().error(f"在 YAML 中找不到指定的 Tag ID: {tag_id}")
        return False

    def timer_callback(self):
        # 檢查氣球資料是否過期有效
        is_target_valid = False
        if self.last_target_time is not None:
            dt = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
            if dt <= self.target_timeout:
                is_target_valid = True

        # ====================================================
        # 自動化狀態機核心分流
        # ====================================================
        if self.state == 'IDLE':
            # 原地安全停懸
            self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0

        elif self.state == 'TAKEOFF':
            # 🆕 新增這個狀態：起飛時強制速度歸零，靜止等待 Tello 升空穩定
            self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0
            self.takeoff_counter += 1
            
            # 20Hz * 4 秒 = 80 次計數
            if self.takeoff_counter >= 80:
                self.state = 'STEP1_SEARCH'
                self.get_logger().info("🛫 Tello 已升空穩定！自動切換至 STEP1_SEARCH 開始找球。")

        elif self.state == 'STEP1_SEARCH':
            if is_target_valid:
                # 瞬間捕捉到氣球，切換至追蹤撞擊狀態
                self.state = 'STEP1_TRACK'
                self.get_logger().info("🎯 發現氣球！切換至 STEP1_TRACK 模式，開始衝撞！")
            else:
                # 沒看到球：原地自轉搜尋
                self.v_x = self.v_y = self.v_z = 0.0
                self.yaw_rate = self.SEARCH_YAW_RATE

        elif self.state == 'STEP1_TRACK':
            if not is_target_valid:
                # 球突然不見，退回搜尋狀態繼續轉圈
                self.state = 'STEP1_SEARCH'
                self.get_logger().warn("⚠️ 氣球丟失，退回 STEP1_SEARCH 狀態重新搜尋。")
            else:
                # 執行原本寫好的 PID 撞球邏輯 (Map to Body 幾何轉換)
                dx_map = self.target_x - self.drone_x
                dy_map = self.target_y - self.drone_y
                dz_map = self.target_z - self.drone_z

                # 轉換至機身座標系
                self.v_x = dx_map * math.cos(self.drone_yaw) + dy_map * math.sin(self.drone_yaw)
                self.v_y = -dx_map * math.sin(self.drone_yaw) + dy_map * math.cos(self.drone_yaw)
                self.v_z = dz_map

                # 乘上 P 增益
                self.v_x *= self.KP_XY
                self.v_y *= self.KP_XY
                self.v_z *= self.KP_Z

                # 轉向控制：機頭主動朝向氣球
                target_yaw = math.atan2(dy_map, dx_map)
                dyaw = target_yaw - self.drone_yaw
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw)) # 最短路徑 [-pi, pi]
                self.yaw_rate = max(min(self.KP_YAW * dyaw, self.MAX_YAW_RATE), -self.MAX_YAW_RATE)

                # 安全限速保護
                self.v_x = max(min(self.v_x, self.MAX_VEL), -self.MAX_VEL)
                self.v_y = max(min(self.v_y, self.MAX_VEL), -self.MAX_VEL)
                self.v_z = max(min(self.v_z, self.MAX_VEL), -self.MAX_VEL)

        elif self.state == 'STEP2_NAVIGATE':
            if self.drone_x is not None and self.goal_x is not None:
                # 1. 計算與目標點 (Tag 前方 30cm) 的世界座標誤差
                dx_map = self.goal_x - self.drone_x
                dy_map = self.goal_y - self.drone_y
                dz_map = self.goal_z - self.drone_z
                
                # 2. 轉換至機身座標系
                self.v_x = dx_map * math.cos(self.drone_yaw) + dy_map * math.sin(self.drone_yaw)
                self.v_y = -dx_map * math.sin(self.drone_yaw) + dy_map * math.cos(self.drone_yaw)
                self.v_z = dz_map
                
                # 3. 乘上 PID 增益
                self.v_x *= self.KP_XY
                self.v_y *= self.KP_XY
                self.v_z *= self.KP_Z
                
                # 4. 角度控制：對準 Tag 的正面
                dyaw = self.goal_yaw - self.drone_yaw
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                self.yaw_rate = max(min(self.KP_YAW * dyaw, self.MAX_YAW_RATE), -self.MAX_YAW_RATE)
                
                # 安全限速
                self.v_x = max(min(self.v_x, self.MAX_VEL), -self.MAX_VEL)
                self.v_y = max(min(self.v_y, self.MAX_VEL), -self.MAX_VEL)
                self.v_z = max(min(self.v_z, self.MAX_VEL), -self.MAX_VEL)

                # 5. 抵達判斷：如果水平誤差 < 10cm 且高度、角度對準，觸發自動降落
                if (dx_map**2 + dy_map**2 < 0.10**2) and (abs(dz_map) < 0.10) and (abs(dyaw) < 0.15):
                    self.state = 'STEP2_LAND'
                    self.get_logger().info("🎉 已精準抵達停機坪上方！準備降落...")

        elif self.state == 'STEP2_LAND':
            self.v_x = self.v_y = self.v_z = self.yaw_rate = 0.0
            self.send_tello_cmd('land')
            self.state = 'IDLE' # 降落後回到待命狀態

        # --- 發布 Twist 控制指令給 Tello ---
        twist = Twist()
        twist.linear.x = float(self.v_x)
        twist.linear.y = float(self.v_y)
        twist.linear.z = float(self.v_z)
        twist.angular.z = float(self.yaw_rate)
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ControlTelloEKF()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 💡 當終端機按下 Ctrl+C 時，會觸發這裡，立刻發送一個速度全為 0 的 Twist
        print("\n🛑 [警告] 偵測到 Ctrl+C 中斷指令！發送懸停指令並關閉節點。")
    finally:
        # 強制無人機原地停懸
        emergency_twist = Twist()
        node.publisher_.publish(emergency_twist)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()