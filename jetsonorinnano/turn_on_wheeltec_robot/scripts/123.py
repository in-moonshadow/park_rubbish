#!/usr/bin/env python
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
from std_srvs.srv import Trigger
from pydub import AudioSegment
from pydub.playback import play
import time
import serial
import threading
from uservo import UartServoManager
import re

finish_event = threading.Event()  # 用于控制结束标志
decision_event = threading.Event()  # 用于控制 decision 的结束
# 定义全局集合用于记录已播放的标签
played_labels = set()

# 定义每个ID对应的音频文件路径
audio_files = {
    "0": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/zhibei.wav',
    "1": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/papper.wav',
    "2": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/bottle.wav',
    "3": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/battery.wav',
    "4": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/yao.wav',
    "5": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/light.wav',
    "6": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/hua.wav',
    "7": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/caiye.wav',
    "8": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/potato.wav',
    "9": '/home/wheeltec/yolov5-v7.0/tensorrtx-yolov5-v7.0/yolov5/park/music/shi.wav'
}

def move_to_goal(x, y, orientation_z, orientation_w):
    """
    将机器人移动到指定的目标位置，并朝向指定的方向。

    Args:
        x (float): 目标位置的 x 坐标。
        y (float): 目标位置的 y 坐标。
        orientation_z (float): 四元数表示的方向向量的 z 分量。
        orientation_w (float): 四元数表示的方向向量的 w 分量。

    Returns:
        bool: 移动是否成功，成功返回 True，否则返回 False。
    """
    # 初始化ROS节点
    rospy.init_node('move_to_goal_node', anonymous=True)

    # 创建MoveBaseAction的客户端，用于发送目标位置
    ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # 等待MoveBaseAction服务器启动
    rospy.loginfo("等待MoveBaseAction服务器启动...")
    ac.wait_for_server()

    # 创建目标位置对象
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # 设置目标位置的坐标
    goal.target_pose.pose.position = Point(x, y, 0)

    # 设置目标位置的方向
    quaternion = Quaternion()
    quaternion.z = orientation_z
    quaternion.w = orientation_w
    goal.target_pose.pose.orientation = quaternion

    # 发送目标位置
    rospy.loginfo("正在向目标位置移动...")
    ac.send_goal(goal)

    # 等待机器人到达目标位置
    ac.wait_for_result()

    # 检查移动结果
    if ac.get_state() == 3:
        rospy.loginfo("机器人成功到达目标位置!")
        return True
    else:
        rospy.loginfo("机器人未能到达目标位置!")
        return False

class angle():
    def __init__(self):
        # 参数配置
        # 角度定义
        self.SERVO_PORT_NAME = '/dev/ttyUSB1'  # 舵机串口号
        self.SERVO_BAUDRATE = 115200  # 舵机的波特率
        # SERVO_ID = 0  # 舵机的ID号
        # 初始化串口
        self.uart = serial.Serial(port=self.SERVO_PORT_NAME, baudrate=self.SERVO_BAUDRATE,
                                  parity=serial.PARITY_NONE, stopbits=1,
                                  bytesize=8, timeout=0)
        self.uservo = UartServoManager(self.uart, is_debug=True)


    def fuwei(self):
        self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(2, 93.2, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 118.6, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -90, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(5, -19.0, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)

    def avoid(self):
        self.uservo.set_servo_angle(1, 0, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(2, 25.1, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 13, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -102.2, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(5, -20.4, interval=2000)  # 设置舵机角度 极速模式

    #开爪
    def open(self):
        self.uservo.set_servo_angle(5, 22.4, interval=1000)  # 设置舵机角度 极速模式
        time.sleep(1)
    #合爪
    def close(self):
        self.uservo.set_servo_angle(5, -25, interval=1000)  # 设置舵机角度 极速模式
        time.sleep(1)
    # 底座角度
    def ang(self):
        angle = 0.0
        angle = x / 10 - 120
        self.uservo.set_servo_angle(1, angle=angle, interval=1000)  # 设置舵机角度 极速模式
    # 抬起
    def up(self):
        #self.uservo.set_servo_angle(1, -63.5, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(2, 55, interval=1000)  # 设置舵机角度 极速模式

        time.sleep(0.8)
        self.uservo.set_servo_angle(2, 15, interval=300)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 0, interval=1180)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -101, interval=680)  # 设置舵机角度 极速模式
        time.sleep(2)
    # 识别下
    def Rshibiexia(self):
        self.uservo.set_servo_angle(1, -90, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.uservo.set_servo_angle(2, 67.6, interval=3000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 109.7, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -80.8, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.uservo.set_servo_angle(5, 22.2, interval=500)  # 设置舵机角度 极速模式
        time.sleep(0.5)
    # 识别上
    def Rshibieshang(self):
        self.uservo.set_servo_angle(1, -90, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.uservo.set_servo_angle(2, 90, interval=3000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 109.7, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -80.8, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.uservo.set_servo_angle(5, 22.2, interval=500)  # 设置舵机角度 极速模式
        time.sleep(0.5)



    # 低放爪
    def Ldown(self):
        self.uservo.set_servo_angle(2, -29.0, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 117.8, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, 33.4, interval=1000)  # 设置舵机角度 极速模式
        time.sleep(2)

    # 高放爪
    def Hdown(self):
        self.uservo.set_servo_angle(2, 68.0, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 117.8, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -45.5, interval=1000)  # 设置舵机角度 极速模式
        time.sleep(2)

    # 低推
    def Lpush(self):
        self.uservo.set_servo_angle(2, -85.2, interval=1500)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 25.8, interval=4500)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -10, interval=3000)  # 设置舵机角度 极速模式
        time.sleep(5)

    # 高推
    def Hpush(self):
        self.uservo.set_servo_angle(2, -13, interval=1500)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 60.7, interval=4500)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -20.7, interval=1500)  # 设置舵机角度 极速模式
        time.sleep(5)

    # 低收
    def Lpull(self):
        self.uservo.set_servo_angle(2, -29.0, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 117.8, interval=1500)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, 33.4, interval=1500)  # 设置舵机角度 极速模式
        time.sleep(2)

    # 高放爪
    def Hpull(self):
        self.uservo.set_servo_angle(2, 55.0, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 117.8, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -45.5, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)

    def Lall(self):
        # arm.fuwei()
        # arm.Rshibiexia()
        arm.ang()
        arm.open()
        arm.Ldown()
        arm.Lpush()
        arm.close()
        arm.Lpull()
        arm.fuwei()

    def Hall(self):
        arm.fuwei()
        arm.Rshibieshang()
        arm.ang()
        arm.open()
        arm.Hdown()
        arm.Hpush()
        arm.close()
        arm.Hdown()
        arm.fuwei()



def parse_detection_message(message):           
    """解析检测消息，提取class_id和边界框坐标."""
    matches = re.findall(
        r"Class:\s*(\d+).*?BBox:\s*\[x:\s*(-?\d+\.?\d*),\s*y:\s*(-?\d+\.?\d*),\s*w:\s*(-?\d+\.?\d*),\s*h:\s*(-?\d+\.?\d*)\]",message)
    if matches:
        first_match = matches[0]
        class_id = first_match[0]
        x = float((float(first_match[1]) + float(first_match[3])) / 2)
        y = float((float(first_match[2]) + float(first_match[4])) / 2)
        return class_id, x, y  # 返回class_id和边界框信息
    return None, None, None  # 解析失败返回None
    
def try_move_to_goal(goal, retries, max_retries, goal_name):
    success = False
    while not success and retries < max_retries:
        result = move_to_goal(*goal)
        if result:
            success = True
            rospy.loginfo(f"成功到达{goal_name}")
        else:
            retries += 1
            rospy.logwarn(f"尝试到{goal_name}失败，正在重试... ({retries}/{max_retries})")
    return success, retries

def decision(id, k):
    max_retries = 3  # 最大重试次数
    retries = 0

    # 定义目标位置和日志信息
    goals = {
        "0": {"name": "可回收", "pos": [
            (3.1754, 2.9712, 0.7015, 0.7026),  # k == 1
            (3.2080, 3.1336, -0.7095, 0.7046)  # k == 2
        ]},
        "1": {"name": "可回收", "pos": [
            (3.1754, 2.9712, 0.7015, 0.7026),  # k == 1
            (3.2080, 3.1336, -0.7095, 0.7046)  # k == 2
        ]},
        "2": {"name": "可回收", "pos": [
            (3.1754, 2.9712, 0.7015, 0.7026),  # k == 1
            (3.2080, 3.1336, -0.7095, 0.7046)  # k == 2
        ]},
        "3": {"name": "有害", "pos": [
            (3.2085, 2.5272, 0.7047, 0.7094),  # k == 1
            (3.2185, 2.5472, -0.7047, 0.7094)  # k == 2
        ]},
        "4": {"name": "有害", "pos": [
            (3.2085, 2.5272, 0.7047, 0.7094),  # k == 1
            (3.2185, 2.5472, -0.7047, 0.7094)  # k == 2
        ]},
        "5": {"name": "有害", "pos": [
            (3.2085, 2.5272, 0.7047, 0.7094),  # k == 1
            (3.2185, 2.5472, -0.7047, 0.7094)  # k == 2
        ]},
        "6": {"name": "厨余", "pos": [
            (3.1957, 2.7769, 0.7095, 0.7046),  # k == 1
            (2.6765, 4.9551, -0.9999, 0.0042)  # k == 2
        ]},
        "7": {"name": "厨余", "pos": [
            (3.1957, 2.7769, 0.7095, 0.7046),  # k == 1
            (2.6765, 4.9551, -0.9999, 0.0042)  # k == 2
        ]},
        "8": {"name": "厨余", "pos": [
            (3.1957, 2.77769, 0.7095, 0.7046),  # k == 1
            (2.6765, 4.9551, -0.9999, 0.0042)  # k == 2
        ]},
        "9": {"name": "其他", "pos": [
            (3.2150, 2.2546, 0.7030, 0.7010),  # k == 1
            (3.2150, 2.2246, -0.7030, 0.7010)  # k == 2
        ]}
    }

    # 检查目标是否存在
    if id in goals:
        goal_name = goals[id]["name"]
        rospy.loginfo(f"开始前往{goal_name}")
        
        # 根据k选择目标坐标
        target_pos = goals[id]["pos"][k - 1]  # k == 1 时选第一个目标，k == 2 时选第二个目标

        # 尝试移动到目标位置
        success, retries = try_move_to_goal(target_pos, retries, max_retries, goal_name)

        rospy.sleep(2)

        # 如果失败，输出错误信息
        if not success:
            rospy.logerr(f"无法到达{goal_name}，已达到最大重试次数.")

    decision_event.set()  # 设置事件，表示 decision 处理完成

def label_callback(msg, i):
    global played_labels
    # 获取消息中的 class_id
    detection_result, x, y = parse_detection_message(msg.data)
    if detection_result is None:
        rospy.logwarn("未能解析检测消息")
        return
    
    if detection_result in played_labels:
        rospy.logwarn(f"检测结果 {detection_result} 已被处理过，跳过处理。")
        return
    
    rospy.loginfo("检测到物品: %s, 中心坐标: (%f, %f)", detection_result, x, y)

    # 播放对应音频
    if detection_result in audio_files and detection_result not in played_labels:
        audio_path = audio_files[detection_result]
        audio = AudioSegment.from_wav(audio_path)
        play(audio)
        played_labels.add(detection_result)  # 记录已播放的标签

        # 判断 i 的值并执行对应的逻辑
        k = 1 if i in {1, 3, 5, 6, 7, 8, 9, 10} else 2  # 根据索引 i 设置 k 值
        action_function = YOUXIA if i in {1, 2, 5, 8, 9, 10} else ZUOXIA  # 根据索引 i 设置动作函数

        action_function(x, y)  # 执行对应的动作函数
        time.sleep(2)  # 等待 2 秒
        decision(detection_result, k)  # 执行决策逻辑

    decision_event.wait()  # 设置事件，表示 decision 处理完成
    finish_event.set()

def get_detection_result(i):
    # 订阅检测结果主题
    rospy.Subscriber("/park_detection/labels", String, callback=lambda msg: label_callback(msg, i))
    rospy.wait_for_service('trigger_detection')
    try:
        trigger_detection = rospy.ServiceProxy('trigger_detection', Trigger)
        response = trigger_detection()
        if response.success:
            rospy.loginfo("Detection triggered successfully.")
        else:
            rospy.logwarn("Failed to trigger detection: %s", response.message)
            return None
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)
        return None

    finish_event.wait()
    rospy.logwarn("开始前往下一个垃圾点")
    decision_event.clear()
    finish_event.clear()
    return None

def move_to_position(i, x, y, orientation_z, orientation_w, max_retries):
    """执行尝试移动到目标位置并处理逻辑"""
    success = False
    retries = 0

    while not success and retries < max_retries:
        if i == 1:
            arm.fuwei()
            rospy.loginfo("机械臂复位")
        
        result = move_to_goal(x, y, orientation_z, orientation_w)
        if result:
            success = True
            rospy.loginfo(f"成功到达第 {i} 个目标位置 ({x}, {y})！")

            if i in {1, 2, 3, 4, 5, 6, 7, 8}:
                arm.Rshibiexia()
                rospy.sleep(3)
                get_detection_result(i)
                arm.youdiu()
                time.sleep(3)
                arm.fuwei()
            elif i in {9, 10}:
                get_detection_result()

        else:
            retries += 1
            rospy.logwarn(
                f"尝试到达第 {i} 个目标位置 ({x}, {y}) 失败，正在重试... ({retries}/{max_retries})"
            )
    return success

if __name__ == '__main__':
    try:
        # 设置目标位置坐标和方向
        goal_positions = [
            ((1.1627, 0.5996), (-0.0461, 0.999)),  # 位置1
            ((2.6765, 4.9151), (-0.9999, 0.0042)),  # 位置2
            ((2.7691, 1.4551), (-0.0116, 0.9999)),  # 塔
            ((1.6382, 4.3440), (-0.7080, 0.7062)),  # 双点
            ((1.2143, 1.7323), (-0.9999, 0.0156)),  # 水右
            ((2.8777, 2.6153), (-0.7030, 0.7010)),  # 水上
            ((0.3347, 2.2780), (-0.7057, 0.7078)),  # 最右2-1
            ((0.2384, 3.1596), (-0.7075, 0.7078)),  # 最右2-2
        ]

        global arm 
        arm= angle()

        # 定义最大重试次数
        max_retries = 3

        # 依次移动到每个目标位置
        for i, (goal_pos, orientation) in enumerate(goal_positions, start=1):
            x, y = goal_pos
            orientation_z, orientation_w = orientation
            
            rospy.loginfo(f"开始尝试移动至第 {i} 个目标位置 ({x}, {y})")
            if not move_to_position(i, x, y, orientation_z, orientation_w, max_retries):
                rospy.logerr(f"无法到达第 {i} 个目标位置 ({x}, {y})，已达到最大重试次数。程序中断。")
                break

            rospy.sleep(2)

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断。")
