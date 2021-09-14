import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_itg3205 import *
from time import *
a = 0
itg3205 = i2c_itg3205(1)
time_start = time()
while(a < 1000):
    a = a + 1
    (itgready, dataready) = itg3205.getInterruptStatus()
    if dataready:
        temp = itg3205.getDieTemperature()
        (x, y, z) = itg3205.getDegPerSecAxes()
       # print ("Temp:" + str (temp ))
       # print ("X:" + str (x ))
       # print ("Y:" + str (y ))
       # print ("Z:" + str (z ))
       # print ("")
print((time() - time_start) / a)
    
