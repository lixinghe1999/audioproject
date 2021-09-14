import sys
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
from time import *
adxl345 = i2c_adxl345(1)
def test_samplerate():
    time_start = time()
    a = 0
    while (a<1000):
        #if adxl345.getInterruptStatus():
        a = a + 1
        (x, y, z) = adxl345.getRawAxes()
    return (time() - time_start) / a

for rate in [0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]:
    adxl345.setdatarate(rate)
    print(adxl345.getdatarate())
    print(test_samplerate())
# with open("acc.txt", "w") as writer:
#     while True:
#         if adxl345.getInterruptStatus():
#             a = a + 1
#             (x, y, z) = adxl345.getAxes()
#             print((time() - time_start)/a)
#             #data = [str(x), str(y), str(z), str(time())]
#             # writer.write(" ".join(data) + '\n')


