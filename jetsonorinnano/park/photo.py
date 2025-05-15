#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import threading

def take_photo():
    # 创建一个VideoCapture对象，参数0表示默认的第一个摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 使用V4L2后端（Linux系统）
    # 设置摄像头分辨率（可根据需要调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    path = "/home/wheeltec/park/photo/1.png"  # 替换为你想要保存的路径和文件名

    # 检查摄像头是否正常打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 定义一个锁，用于保护共享资源的访问
    lock = threading.Lock()

    def capture_and_save():
        # 获取锁
        lock.acquire()

        # 从摄像头捕获一帧图像
        ret, frame = cap.read()

        if ret:
            # 保存图片
            cv2.imwrite(path, frame)
            print("成功保存图片到: {}".format(path))
        else:
            print("无法读取摄像头帧")

        # 释放锁
        lock.release()

    # 开启线程进行图像捕获和保存
    thread = threading.Thread(target=capture_and_save)
    thread.start()

    # 等待线程完成
    thread.join()

    # 关闭摄像头
    cap.release()

# 调用函数拍照并保存
take_photo()

