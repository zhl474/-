import serial

ser = serial.Serial()
ser.port = '/dev/ttyUSB0'  # 设置串口号
ser.baudrate = 115200  # 设置波特率
ser.open()  # 打开串口