import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
from i2c_itg3205 import *
from i2c_hmc5883l import *
from time import *

a = 0
adxl345 = i2c_adxl345(1)
hmc5883l = i2c_hmc5883l(1)
itg3205 = i2c_itg3205(1)
hmc5883l.setContinuousMode()
hmc5883l.setDeclination(3, 5)

time_start = time()
while True:
    a = a + 1
    print((time()-time_start) / a)
    # save acc data
    adxl345.save()
    print(hmc5883l)
    (itgready, dataready) = itg3205.getInterruptStatus()
    if dataready:
        temp = itg3205.getDieTemperature()
        (x, y, z) = itg3205.getDegPerSecAxes()
        # print ("Temp:" + str (temp ))
        # print ("X:" + str (x ))
        # print ("Y:" + str (y ))
        # print ("Z:" + str (z ))
        # print ("")
