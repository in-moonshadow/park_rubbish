#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion
from vision_msgs.msg import Detection2DArray
from pydub import AudioSegment
from pydub.playback import play
import os
import threading
import time
import serial
from uservo import UartServoManager
import math


# --- 全局变量 ---
spoken_ids = set()

id_to_audio = {
    0: '/home/wheeltec/park/src/park/music/zhibei.wav',
    1: '/home/wheeltec/park/src/park/music/papper.wav',
    2: '/home/wheeltec/park/src/park/music/bottle.wav',
    3: '/home/wheeltec/park/src/park/music/battery.wav',
    4: '/home/wheeltec/park/src/park/music/yao.wav',
    5: '/home/wheeltec/park/src/park/music/light.wav',
    6: '/home/wheeltec/park/src/park/music/hua.wav',
    7: '/home/wheeltec/park/src/park/music/caiye.wav',
    8: '/home/wheeltec/park/src/park/music/potato.wav',
    9: '/home/wheeltec/park/src/park/music/shi.wav'
}

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
        angle = self.uservo.query_servo_angle(1)  # 查询对应 id 的舵机角度
        if angle <= 0:
            self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        else:
            self.uservo.set_servo_angle(1, 95, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(2, 93.2, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 118.6, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -90, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(5, -19.0, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)


    def Rfuwei(self):
        self.uservo.set_servo_angle(1, 95.2, interval=2000)  # 设置舵机角度 极速模式
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
        self.uservo.set_servo_angle(5, -22, interval=1000)  # 设置舵机角度 极速模式
        time.sleep(1)
    # 底座角度
    def ang(self,x):
        o_postion = self.uservo.query_servo_angle(1)
        if o_postion <= 0:
            angle = 0.0
            angle = x / 10 - 120
            self.uservo.set_servo_angle(1, angle=angle, interval=1000)  # 设置舵机角度 极速模式
        else:
            angle = 0.0
            angle = x / 10 + 65
            self.uservo.set_servo_angle(1, angle=angle, interval=1000)  # 设置舵机角度 极速模式

    # 抬起
    def up(self):
        #self.uservo.set_servo_angle(1, -63.5, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(2, 55, interval=1200)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -70, interval=800)  # 设置舵机角度 极速模式
        time.sleep(0.8)
        self.uservo.set_servo_angle(2, 15, interval=600)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 0, interval=800)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -101, interval=680)  # 设置舵机角度 极速模式
        time.sleep(2)

    # 从抬起动作收回至复位
    def down(self):
        self.uservo.set_servo_angle(2, 93.2, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 118.6, interval=1000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -90, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)

    def turnL(self):
        self.uservo.set_servo_angle(2, 55, interval=1200)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -70, interval=800)  # 设置舵机角度 极速模式
        time.sleep(0.8)
        self.uservo.set_servo_angle(2, 15, interval=600)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 0, interval=800)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -101, interval=680)  # 设置舵机角度 极速模式
        time.sleep(2)
        self.uservo.set_servo_angle(1, 95, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.fuwei()

    def turnR(self):
        self.uservo.set_servo_angle(2, 55, interval=1200)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -70, interval=800)  # 设置舵机角度 极速模式
        time.sleep(0.8)
        self.uservo.set_servo_angle(2, 15, interval=600)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 0, interval=800)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -101, interval=680)  # 设置舵机角度 极速模式
        time.sleep(2)
        self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.fuwei()


    # 识别下
    def Rshibiexia(self):
        angle = self.uservo.query_servo_angle(1)  # 查询对应 id 的舵机角度
        if angle >= 0:
            self.turnR()
        self.uservo.set_servo_angle(1, -90, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.uservo.set_servo_angle(2, 67.6, interval=3000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 109.7, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -80.8, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(3)
        self.uservo.set_servo_angle(5, 22.2, interval=500)  # 设置舵机角度 极速模式
        time.sleep(0.5)

    def Lshibiexia(self):
        angle = self.uservo.query_servo_angle(1)  # 查询对应 id 的舵机角度
        if angle <= 0:
            self.turnL()
        self.uservo.set_servo_angle(1, 95, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)
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
        self.uservo.set_servo_angle(4, 28.4, interval=1000)  # 设置舵机角度 极速模式
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

    # 33.7 0 -25.9
    # 扔1
    def reng1(self):
        self.up()
        time.sleep(1)
        self.uservo.set_servo_angle(1, 33.7, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)
        self.open()
        self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)
        self.close()
        self.down()

    def reng2(self):
        self.up()
        self.uservo.set_servo_angle(1, 0, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2.5)
        self.open()
        self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)
        self.close()
        self.down()

    def reng3(self):
        self.up()
        self.uservo.set_servo_angle(1, -25.9, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2.5)
        self.open()
        self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)
        self.close()
        self.down()

    def LowAllL(self,x,y):
        self.fuwei()
        angle = self.uservo.query_servo_angle(1)
        if angle <= 0:
            self.turnL()
        self.ang(x)
        self.open()
        self.Ldown()
        self.set_servos_based_on_distance(image_x=x, image_y=y)
        self.close()
        self.Lpull()
        self.fuwei()

    def LowAllR(self,x,y):
        self.fuwei()
        angle = self.uservo.query_servo_angle(1)
        if angle > 0:
            self.turnR()
        self.ang(x)
        self.open()
        self.Ldown()
        self.set_servos_based_on_distance(image_x=x, image_y=y)
        self.close()
        self.Lpull()
        self.fuwei()

    def HighAll(self):
        self.fuwei()
        self.ang()
        self.open()
        self.Hdown()
        self.Hpush()
        self.close()
        self.Hdown()
        self.fuwei()

    def set_servos_based_on_distance(self, image_x, image_y, polar_origin_x=320, polar_origin_y=1200):
        # 第一步：计算极坐标距离
        distance = math.sqrt((image_x - polar_origin_x) ** 2 + (image_y - polar_origin_y) ** 2)
        # 第二步：定义已知的距离和对应的角度/间隔
        distance_1 = 813
        distance_2 = 1026
        # 在 distance_1 时的角度和间隔
        angle_2_1 = -29.0
        angle_3_1 = 117.8
        # angle_4_1 = 33.4
        angle_4_1 = 28.4
        interval_2_1 = 2000
        interval_3_1 = 1000
        interval_4_1 = 1000
        # 在 distance_2 时的角度和间隔
        angle_2_2 = -85.2
        angle_3_2 = 25.8
        # angle_4_2 = -10.0
        angle_4_2 = -5.0
        interval_2_2 = 1500
        interval_3_2 = 4500
        interval_4_2 = 3000
        # 第三步：确保距离在定义的范围内
        # distance < distance_1 or distance > distance_2:
        #  raise ValueError("距离必须在 813 到 1026 之间")
        # 第四步：计算标准化距离
        t = (distance - distance_1) / (distance_2 - distance_1)
        # 第五步：使用线性插值计算角度
        servo_angle_2 = angle_2_1 + t * (angle_2_2 - angle_2_1)
        servo_angle_3 = angle_3_1 + t * (angle_3_2 - angle_3_1)
        servo_angle_4 = angle_4_1 + t * (angle_4_2 - angle_4_1)
        # 第六步：使用线性插值计算间隔
        servo_interval_2 = interval_2_1 + t * (interval_2_2 - interval_2_1)
        servo_interval_3 = interval_3_1 + t * (interval_3_2 - interval_3_1)
        servo_interval_4 = interval_4_1 + t * (interval_4_2 - interval_4_1)
        # 第七步：设置舵机角度和间隔
        self.uservo.set_servo_angle(2, servo_angle_2, interval=int(servo_interval_2))
        self.uservo.set_servo_angle(3, servo_angle_3, interval=int(servo_interval_3))
        self.uservo.set_servo_angle(4, servo_angle_4, interval=int(servo_interval_4))
        # 可选：根据需要调整睡眠时间
        time.sleep(2)
        return distance  # 返回计算出的极坐标距离

def play_audio(file_path):
    """
    播放指定的音频文件
    """
    if os.path.exists(file_path):
        rospy.loginfo("Playing audio file: %s", file_path)
        audio = AudioSegment.from_file(file_path)
        play(audio)
    else:
        rospy.logwarn("Audio file not found: %s", file_path)

def detection_once(timeout=10):
    detections_received = threading.Event()
    processed = False  # 标志位，避免多次处理

    def callback(data):
        nonlocal processed
        if processed:
            return  # 已处理过，忽略后续消息

        rospy.loginfo("Received detection results:")
        if not data.detections:
            rospy.logwarn("当前检测消息中无目标")
            processed = True
            detections_received.set()
            center_x = 0.0
            rospy.set_param('/target_center_x', center_x)
            return

        # 取第一个检测目标中心点，写参数服务器
        detection = data.detections[0]
        center_x = detection.bbox.center.x
        center_y = detection.bbox.center.y

        rospy.set_param('/target_center_x', center_x)
        rospy.set_param('/target_center_y', center_y)
        rospy.loginfo(f"Set ROS params: /target_center_x={center_x}, /target_center_y={center_y}")

        # 播放音频且防止重复播放
        for det in data.detections:
            for result in det.results:
                class_id = result.id
                if class_id not in spoken_ids:
                    spoken_ids.add(class_id)
                    if class_id in id_to_audio:
                        play_audio(id_to_audio[class_id])
                    else:
                        rospy.logwarn(f"No audio file for ClassID {class_id}")

        processed = True
        detections_received.set()

    sub = rospy.Subscriber("/yolo_detections", Detection2DArray, callback)
    rospy.loginfo("Waiting for detection message on /yolo_detections...")

    if not detections_received.wait(timeout):
        rospy.logwarn(f"Detection message not received in {timeout} seconds.")
    else:
        rospy.loginfo("Detection processed.")

    sub.unregister()


def move_to_goal(ac, x, y, orientation_z, orientation_w, timeout=60):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position = Point(x, y, 0)
    goal.target_pose.pose.orientation = Quaternion(0, 0, orientation_z, orientation_w)

    rospy.loginfo(f"Sending goal: x={x}, y={y}, orientation_z={orientation_z}, orientation_w={orientation_w}")
    ac.send_goal(goal)

    finished_within_time = ac.wait_for_result(rospy.Duration(timeout))
    if not finished_within_time:
        ac.cancel_goal()
        rospy.logwarn("Goal timeout, canceling goal.")
        return False
    else:
        state = ac.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Reached the goal successfully!")
            return True
        else:
            rospy.logwarn(f"Failed to reach goal. State: {state}")
            return False

def main():
    rospy.init_node("move_to_goal_and_detect_node", anonymous=True)
    audio = AudioSegment.from_wav('/home/wheeltec/park/src/park/music/bobao.wav')
    play(audio)
    arm = angle()
    arm.fuwei()

    ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    if not ac.wait_for_server(rospy.Duration(20)):
        rospy.logerr("move_base action server not available!")
        return

    goal_positions = [
        ((0.98, 0.707), (0.0164, 0.999)), #1
            #((1.610, 0.633), (-0.003, 0.999)), 2
        ((2.371, 1.509), (0.0164, 0.999)),

        ((2.973, 2.401), (0.711, 0.702)),
	    
        ((3.112,4.774), (0.012, 0.999)),
        ((2.455,4.772), (-0.999, 0.009)),#
        ((0.357,4.352), (-0.709, 0.705)),#
        ((0.210,3.275), (-0.709, 0.705)),
        ((0.375,2.541), (-0.709, 0.705)),
        ((0.525,2.401), (-0.709, 0.705)),
    ]

    max_retries = 5

    for i, (pos, orientation) in enumerate(goal_positions, start=1):
        x, y = pos
        orientation_z, orientation_w = orientation
        success = False
        retries = 0

        while not success and retries < max_retries and not rospy.is_shutdown():
            success = move_to_goal(ac, x, y, orientation_z, orientation_w)
            if success:
                rospy.loginfo(f"Arrived at goal {i}: {pos}")
                if i in [1,2,5,7,8]:
                    arm.Rshibiexia()
                    time.sleep(1)
                    detection_once(timeout=10)  # 单次检测
                    target_x = rospy.get_param('/target_center_x')
                    target_y =rospy.get_param('/target_center_y')
                    if target_x!=0.0:
                        arm.LowAllR(target_x,target_y)
                        arm.reng1()
                    else:
                        arm.fuwei()
                if i in [3,4,6,9]:
                    arm.Lshibiexia()
                    time.sleep(2)
                    detection_once(timeout=10)  # 单次检测
                    target_x = rospy.get_param('/target_center_x')
                    target_y =rospy.get_param('/target_center_y')
                    if target_x!=0.0:
                        arm.LowAllL(target_x,target_y)
                        arm.reng1()
                    else:
                        arm.fuwei()
            else:
                retries += 1
                rospy.logwarn(f"Retrying to reach goal {i} ({retries}/{max_retries})...")
                rospy.sleep(2)

        if not success:
            rospy.logerr(f"Failed to reach goal {i} after {max_retries} retries. Stopping mission.")
            break

        rospy.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted and shutting down.")

