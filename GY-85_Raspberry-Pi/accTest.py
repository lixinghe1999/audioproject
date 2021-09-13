import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
from time import *
adxl345 = i2c_adxl345(1)
with open("acc.txt", "w") as writer:
    while True:
        if adxl345.getInterruptStatus():
            (x, y, z) = adxl345.getAxes()
            data = [str(x), str(y), str(z), str(time())]
            writer.write(" ".join(data) + '\n')

