import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
from time import *
a = 0
adxl345 = i2c_adxl345(1)
time_start = time()
adxl345.save()
