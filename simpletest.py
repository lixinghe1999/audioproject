import time
import board

import adafruit_adxl34x

i2c = board.I2C()
accelerometer = adafruit_adxl34x.ADXL345(i2c)
a = 0
accelerometer._write_register_byte(accelerometer._REG_BW_RATE, adafruit_adxl34x.DataRate.RATE_3200_HZ)
time_start = time.time()
while True:
    a = a + 1
    print((time.time() - time_start) / a)
    print("%f %f %f" % accelerometer.acceleration)
