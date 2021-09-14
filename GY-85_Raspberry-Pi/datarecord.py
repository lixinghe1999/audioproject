import sys
import threading
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
from i2c_itg3205 import *
from i2c_hmc5883l import *
import time


hmc5883l = i2c_hmc5883l(1)
hmc5883l.setContinuousMode()
hmc5883l.setDeclination(3, 5)


compasswriter = open('compass.txt', 'w')
time_start = time.time()
def acc_save(time_start):
    a = 0
    adxl345 = i2c_adxl345(1)
    adxl345.setdatarate(0x0F)
    accwriter = open('acc.txt', 'w')
    while (a < 1000):
        if adxl345.getInterruptStatus():
            a = a + 1
            (x1, y1, z1) = adxl345.getRawAxes()
            accwriter.write(str(x1) + ' ' + str(y1) + ' ' + str(z1) + '\n')
    print('acc 1000')
    print((time.time() - time_start) / a)
def gyro_save(time_start):
    b = 0
    itg3205 = i2c_itg3205(1)
    gyrowriter = open('gyro.txt', 'w')
    while (b < 1000):
        itgready, dataready = itg3205.getInterruptStatus()
        if dataready:
            (x2, y2, z2) = itg3205.getDegPerSecAxes()
            gyrowriter.write(str(x2) + ' ' + str(y2) + ' ' + str(z2) + '\n')
            b = b + 1
    print('gyro 1000')
    print((time.time() - time_start) / b)
thread1 = threading.Thread(target = acc_save, args = (time_start))
thread2 = threading.Thread(target = gyro_save, args =(time_start))
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print('end')


# for i in range(10000):
#     if ( i % 200 == 0):
#         (x, y, z) = hmc5883l.getAxes()
#         compasswriter.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
#         c = c + 1
#     if adxl345.getInterruptStatus():
#         (x1, y1, z1) = adxl345.getRawAxes()
#         accwriter.write(str(x1) + ' ' + str(y1) + ' ' + str(z1) + '\n')
#         a = a + 1
#     else:
#         itgready, dataready = itg3205.getInterruptStatus()
#         if dataready:
#             (x2, y2, z2) = itg3205.getDegPerSecAxes()
#             gyrowriter.write(str(x2) + ' ' + str(y2) + ' ' + str(z2) + '\n')
#             b = b + 1
# time_pass = time.time() - time_start
# print(a/time_pass, b/time_pass, c/time_pass)
