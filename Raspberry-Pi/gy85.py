import sys
from i2c_adxl345 import *
from i2c_itg3205 import *
from i2c_hmc5883l import *
sys.path.insert(1,'/home/pi/audioproject/Raspberry-Pi/i2clibraries')
def gy85_accsave(num, port):
    a = 0
    adxl345 = i2c_adxl345(port)
    adxl345.setdatarate(0x0F)
    accwriter = open('acc.txt', 'w')
    acc = ''
    time_start = time.time()
    while (a < num):
        if adxl345.getInterruptStatus():
            a = a + 1
            (x1, y1, z1) = adxl345.getAxes()
            acc = acc + str(x1) + ' ' + str(y1) + ' ' + str(z1) + ' ' + str(time.time()) + '\n'
        else:
            accwriter.write(acc)
            acc = ''
    print('port:', port, a / (time.time() - time_start))

def gy85_gyrosave(num):
    b = 0
    itg3205 = i2c_itg3205(1)
    gyrowriter = open('gyro.txt', 'w')
    time_start = time.time()
    while (b < num):
        itgready, dataready = itg3205.getInterruptStatus()
        if dataready:
            (x2, y2, z2) = itg3205.getDegPerSecAxes()
            gyrowriter.write(str(x2) + ' ' + str(y2) + ' ' + str(z2) + ' ' + str(time.time()) + '\n')
            b = b + 1
    print('gyro', b/(time.time() - time_start))
def gy85_compasssave(num):
    c = 0
    hmc5883l = i2c_hmc5883l(1)
    hmc5883l.setContinuousMode()
    hmc5883l.setDeclination(3, 5)
    compasswriter = open('compass.txt', 'w')
    while (c < num):
        (x3, y3, z3) = hmc5883l.getAxes()
        compasswriter.write(str(x3) + ' ' + str(y3) + ' ' + str(z3) + ' ' + str(time.time()) + '\n')
        c = c + 1
        time.sleep(0.07)
if __name__ == "__main__":
    gy85_accsave(15000, 1)