#!/usr/bin/env python
import rospy
from vision_msgs.msg import Detection2DArray
from pydub import AudioSegment
from pydub.playback import play
import os

# 用于记录已经播报过的ID
spoken_ids = set()

# ID与音乐文件路径的映射
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

def detection_callback(data):
    """
    回调函数，处理订阅到的检测结果消息
    """
    rospy.loginfo("Received detection results:")
    for detection in data.detections:
        for result in detection.results:
            class_id = result.id
            confidence = result.score
            rospy.loginfo("Detection: ClassID=%d, Confidence=%.2f", class_id, confidence)
            if class_id not in spoken_ids:
                spoken_ids.add(class_id)
                if class_id in id_to_audio:
                    play_audio(id_to_audio[class_id])
                else:
                    rospy.logwarn("No audio file associated with ClassID %d", class_id)

def listener():
    """
    初始化节点并订阅检测结果话题
    """
    rospy.init_node('detection_listener', anonymous=True)
    rospy.Subscriber("/yolo_detections", Detection2DArray, detection_callback)
    rospy.loginfo("Listening for detection results on /yolo_detections")
    rospy.spin()  # 保持节点运行，直到被手动停止

if __name__ == '__main__':
    listener()
