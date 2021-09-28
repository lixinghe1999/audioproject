import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
adxl345_1 = i2c_adxl345(1)
adxl345_2 = i2c_adxl345(6)
while True:
    print (adxl345_1)
    time.sleep(0.5)
    print(adxl345_2)
    time.sleep(0.5)
