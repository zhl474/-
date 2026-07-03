#!/home/zhl/fr3env/fr3env/bin/python
import rospy
import threading
import time
import warnings
import serial
from akai_fr import AkaiFr, AkaiElectricSucker

from control.srv import arm,armResponse,motor,motorResponse,suck,suckResponse

class Control:
    def __init__(self):
        self.arm  = AkaiFr()
        self.speed = 80
        self.arm.set_speed(self.speed)
        tool_id = 1
        tcf = [0, 0, 0, 0, 0, 0.0]
        self.arm.set_tcf(tool_id, tcf)
        self.sucker = AkaiElectricSucker(self.arm)    # 创建电子吸盘对象

        self.ser = serial.Serial()
        self.ser.port = '/dev/ttyUSB0'  # 设置串口号
        self.ser.baudrate = 115200  # 设置波特率
        self.ser.open()  # 打开串口

        self.arm_server = rospy.Service("arm_control",arm,self.arm_control)
        self.set_servo_angle_server = rospy.Service("motor_control",motor,self.motor_control)
        self.suck_server = rospy.Service("suck_control",suck,self.suck_control)

        # monitor_thread = threading.Thread(target=self.monitor_serial, daemon=True)
        # monitor_thread.start()
        rospy.loginfo("机械臂/吸盘/舵机控制节点启动")
    def suck_in(self):  
        self.sucker.set_solenoid_valve(False)   # 电磁阀关闭
        self.sucker.set_pump_motor(True)        # 气泵电机开启
    def suck_out(self):
        self.sucker.set_solenoid_valve(True)    # 电磁阀开启
        self.sucker.set_pump_motor(True)        # 气泵电机开启
    def sucker_off(self):
        self.sucker.set_solenoid_valve(False)   # 电磁阀关闭
        self.sucker.set_pump_motor(False)       # 气泵电机关闭
    def suck_control(self,req):
        if(req.state==0):
            self.suck_in()
        elif(req.state==1):
            self.suck_out()
        elif(req.state==2):
            self.sucker_off()
        else:
            rospy.logerr("无效输入")
        return suckResponse(1)
    
        
    def write_to_serial(self, data):
        """向串口发送数据"""
        try:
            self.ser.write(data.encode())
            print(f"Sent: {data}")
        except Exception as e:
            print(f"Failed to send data: {e}")
    def motor_control(self,req):
        angle = req.angle
        if(angle<0):
            angle=0
            warnings.warn("旋转角小于0")
        if(angle>360):
            angle=360
            warnings.warn("旋转角大于360")
        str_angle=str(angle)
        str_angle=str_angle+"E"
        self.write_to_serial(str_angle)
        rospy.set_param('/motor_done', 0)
        # print("舵机开始运行")
        resp = motorResponse(1)
        return resp
    def monitor_serial(self):
        """监控串口,收到数据就将response_flag置1"""    
        while True:
            if self.ser.in_waiting > 0:
                # 有数据来了
                data = self.ser.read_all()  # 读取所有数据
                print("收到舵机数据:", data)
                if data == b'1':
                    rospy.set_param('/motor_done', 1)
                self.ser.reset_input_buffer()  # 清空接收缓冲区
            time.sleep(0.1)  # 避免CPU占用过高


    def arm_control(self,req):
        pose = req.pose
        if req.pose[2]<157:
            pose[2] = 157
            rospy.logerr("高度小于160过低,拒绝运动")
            print("目标高度",req.pose)
        # else:
        self.arm.set_speed(req.speed)
        self.arm.arm.MoveL(pose,tool=0,user=0, vel=req.speed)
        resp = armResponse(1)
        return resp


if __name__ == "__main__":
    # 2.初始化 ROS 节点
    rospy.init_node("control_node")
    # 3.创建服务对象
    Controller = Control()
    # 4.回调函数处理请求并产生响应
    # 5.spin 函数
    rospy.spin()