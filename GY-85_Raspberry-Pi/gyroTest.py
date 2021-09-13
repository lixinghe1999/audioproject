import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_itg3205 import *
from time import *
a = 0
itg3205 = i2c_itg3205(1)
time_start = time()
while True:
    (itgready, dataready) = itg3205.getInterruptStatus ()
    a = a + 1
    print((time()-time_start)/a)   
    if dataready:
        #a = a + 1
        #print((time() - time_start)/a) 
        temp = itg3205.getDieTemperature ()
        (x, y, z) = itg3205.getDegPerSecAxes ()
       # print ("Temp:" + str (temp ))
       # print ("X:" + str (x ))
       # print ("Y:" + str (y ))
       # print ("Z:" + str (z ))
       # print ("")
   # sleep (1)
    
