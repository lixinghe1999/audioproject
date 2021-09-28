import sys
import time

sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
#adxl345_1 = i2c_adxl345(1)

adxl345_2 = i2c_adxl345(6)
adxl345_2.setdatarate(0x0F)
i = 0
time_start = time.time()
while (i<1000):
    i = i + 1
    #print (adxl345_1)
    (x1, y1, z1) = adxl345_2.getAxes()
    acc = str(x1) + ' ' + str(y1) + ' ' + str(z1) + ' ' + str(time.time()) + '\n'
print(1000/(time.time() - time_start))


