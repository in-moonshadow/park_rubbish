# coding=utf-8
import time
import struct
import serial
from uservo import UartServoManager

class angle():
    def __init__(self):
        # 参数配置
        # 角度定义
        self.SERVO_PORT_NAME = '/dev/ttyUSB2'  # 舵机串口号
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
    
    def Lshibiexia(self):
        self.uservo.set_servo_angle(1, 90, interval=2000)  # 设置舵机角度 极速模式
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
    
    def chaxun(self):
        print("-> {}".format(self.uservo.query_servo_angle(1)))
        print("-> {}".format(self.uservo.query_servo_angle(2)))
        print("-> {}".format(self.uservo.query_servo_angle(3)))
        print("-> {}".format(self.uservo.query_servo_angle(4)))
    
    def reng1(self):
        self.up()
        time.sleep(1)
        self.uservo.set_servo_angle(1, 33.7, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2.5)
        self.open()
        self.uservo.set_servo_angle(1, -91.2, interval=2000)  # 设置舵机角度 极速模式
        time.sleep(2)
        # self.fuwei()

    def up(self):
        #self.uservo.set_servo_angle(1, -63.5, interval=2000)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(2, 55, interval=1000)  # 设置舵机角度 极速模式

        time.sleep(0.8)
        self.uservo.set_servo_angle(2, 15, interval=300)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(3, 0, interval=1180)  # 设置舵机角度 极速模式
        self.uservo.set_servo_angle(4, -101, interval=680)  # 设置舵机角度 极速模式
        time.sleep(2)


x = 468.5
y = 217
# 640*480

arm = angle()
arm.fuwei()
# arm.up()
# time.sleep(3)
# # arm.fuwei()
# arm.Lshibiexia()
# time.sleep(1)
# # arm.reng1()
# arm.chaxun()
#arm.open()
#arm.down()
#arm.push()
#arm.close()
#arm.pull()
#time.sleep(2)

#arm.ang()
#arm.avoid()
#arm.up()
# time.sleep(3)
# arm.hezhuaA()
# time.sleep(3)
# arm.fuwei()