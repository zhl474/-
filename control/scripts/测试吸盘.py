import rospy
import threading
import serial
from akai_fr import AkaiFr, AkaiElectricSucker


arm  = AkaiFr()
arm.set_speed(40)
sucker = AkaiElectricSucker(arm)

def suck_in():  
    sucker.set_solenoid_valve(False)   # 电磁阀关闭
    sucker.set_pump_motor(True)        # 气泵电机开启
def suck_out():
    sucker.set_solenoid_valve(True)    # 电磁阀开启
    sucker.set_pump_motor(True)        # 气泵电机开启
def sucker_off():
    sucker.set_solenoid_valve(False)   # 电磁阀关闭
    sucker.set_pump_motor(False)       # 气泵电机关闭
def suck_control(req):
    if(req==0):
        suck_in()
    elif(req==1):
        suck_out()
    elif(req==2):
        sucker_off()

suck_control(2)