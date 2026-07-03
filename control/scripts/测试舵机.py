import time
import serial
import threading

ser = serial.Serial()
ser.port = '/dev/ttyUSB0'  # 设置串口号
ser.baudrate = 115200  # 设置波特率
ser.open()  # 打开串口

flag = 1

def write_to_serial( data):
    """向串口发送数据"""
    try:
        ser.write(data.encode())
        print(f"Sent: {data}")
    except Exception as e:
        print(f"Failed to send data: {e}")
def motor_control(angle):
    global flag
    if(angle<0):
        angle=0
    if(angle>360):
        angle=360
    str_angle=str(angle)
    str_angle=str_angle+"E"
    write_to_serial(str_angle)
    flag = 0
    print("舵机开始运行")

def monitor_serial():
    global flag
    """监控串口,收到数据就将response_flag置1"""    
    while True:
        if ser.in_waiting > 0:
            # 有数据来了
            data = ser.read_all()  # 读取所有数据
            print("收到舵机数据:", data)
            if data == b'1':
                flag = 1
            ser.reset_input_buffer()  # 清空接收缓冲区
        # time.sleep(0.1)  # 避免CPU占用过高

monitor_thread = threading.Thread(target=monitor_serial, daemon=True)
monitor_thread.start()

motor_done = 1
def timer_callback():
    global motor_done
    motor_done = 1
    print("定时器结束，标志位已设为1")

while True:
    angle_velocity = 180
    timer = threading.Timer(360/angle_velocity, timer_callback)
    timer.start()
    motor_control(0)
    motor_done = 0
    while not motor_done:
        pass
    motor_control(360)
    motor_done = 0
    timer = threading.Timer(360/angle_velocity, timer_callback)
    timer.start()
    while not motor_done:
        pass
    # time.sleep(0.5)