#!/usr/bin/env python
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from geometry_msgs.msg import Pose, Point, Quaternion

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

if __name__ == '__main__':
    try:
        # 设置目标位置坐标和方向
        goal_positions = [
            ((1.05, 0.707), (0.0164, 0.999)), #1
            #((1.610, 0.633), (-0.003, 0.999)), 2
            ((2.371, 1.509), (0.0164, 0.999)),

	        ((2.913, 2.401), (0.711, 0.702)),
	    
	        ((3.162,4.734), (-0.999, 0.012)),
	        ((2.455,4.702), (-0.999, 0.009)),#
            ((0.357,4.402), (-0.709, 0.705)),#
            ((0.300,3.275), (-0.709, 0.705)),
            ((0.375,2.541), (-0.709, 0.705)),
            ((0.525,2.401), (-0.709, 0.705)),
        ]

        # 定义重试次数
        max_retries = 5  # 可以根据需要调整最大重试次数

        # 依次移动到每个目标位置
        for i, (goal_pos, orientation) in enumerate(goal_positions, start=1):
            x, y = goal_pos
            orientation_z, orientation_w = orientation
            success = False
            retries = 0
            while not success and retries < max_retries:
                result = move_to_goal(x, y, orientation_z, orientation_w)
                if result:
                    success = True
                    rospy.loginfo(f"成功到达第 {i} 个目标位置 {goal_pos}!")
                    rospy.sleep(3)
                else:
                    retries += 1
                    rospy.logwarn(f"尝试到达第 {i} 个目标位置 {goal_pos} 失败，正在重试... ({retries}/{max_retries})")
            rospy.sleep(2)
            
            if not success:
                rospy.logerr(f"无法到达第 {i} 个目标位置 {goal_pos}，已达到最大重试次数.")
                break  # 如果达到了最大重试次数，停止尝试后续的目标点

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断.")
