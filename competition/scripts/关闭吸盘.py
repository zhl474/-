from akai_fr import AkaiFr, AkaiElectricSucker

arm  = AkaiFr()
sucker = AkaiElectricSucker(arm)
def suck_in():
    # 电磁阀关闭
    sucker.set_solenoid_valve(False)
    # 气泵电机开启
    sucker.set_pump_motor(True)
def suck_out():
    # 电磁阀开启
    sucker.set_solenoid_valve(True)
    # 气泵电机开启
    sucker.set_pump_motor(True)
def suck_off():
    # 电磁阀关闭
    sucker.set_solenoid_valve(False)
    #气泵电机关闭
    sucker.set_pump_motor(False)

# suck_in()
suck_off()