import time
import board

import adafruit_adxl34x

i2c = board.I2C()
accelerometer = adafruit_adxl34x.ADXL345(i2c)
#a = 0
accelerometer._write_register_byte(adafruit_adxl34x._REG_BW_RATE, adafruit_adxl34x.DataRate.RATE_3200_HZ)
time_start = time.time()
for a in range(1000):
    print("%f %f %f" % accelerometer.acceleration)
print((time.time() - time_start) / a)
