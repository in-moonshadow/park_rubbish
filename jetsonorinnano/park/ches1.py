import cv2
import os
import datetime
from pynput.keyboard import Listener

def save_image(frame, folder_path):
    """
    保存图片到指定文件夹
    :param frame: 当前帧
    :param folder_path: 保存路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 如果文件夹不存在，创建文件夹

    # 生成图片文件名，以当前时间戳命名，避免重复
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"image_{timestamp}.jpg"
    filepath = os.path.join(folder_path, filename)

    cv2.imwrite(filepath, frame)  # 保存图片
    print(f"图片已保存到 {filepath}")

def on_press(key):
    """
    按键事件处理函数
    """
    global cap, folder_path, running

    try:
        print(f"检测到按键: {key.char}")  # 调试信息
        if key.char == 's':  # 按下 's' 键保存图片
            ret, frame = cap.read()
            if ret:
                save_image(frame, folder_path)
            else:
                print("无法读取摄像头帧")
        elif key.char == 'q':  # 按下 'q' 键退出程序
            running = False
            print("退出程序")
    except AttributeError:
        print(f"检测到特殊按键: {key}")  # 调试信息

def main():
    global cap, folder_path, running

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("摄像头已打开，按 's' 键保存图片，按 'q' 键退出...")

    # 指定保存图片的文件夹
    folder_path = "saved_images"

    # 初始化运行标志
    running = True

    # 创建键盘监听器
    listener = Listener(on_press=on_press)
    listener.start()

    # 主循环，保持程序运行
    while running:
        pass  # 等待按键事件

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()