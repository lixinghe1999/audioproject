import sys
sys.path.insert(1,'/home/pi/audioproject/Raspberry-Pi/i2clibraries')
from i2c_hmc5883l import *
from time import *
hmc5883l = i2c_hmc5883l(1)
i = 1
time_start = time()
hmc5883l.setContinuousMode ()
hmc5883l.setDeclination (3,5)
while True:
    print(hmc5883l)
    i = i+1
    print((time() - time_start) / a)
