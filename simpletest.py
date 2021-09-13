import time
import board
import adafruit_adxl34x

i2c = board.I2C()  # uses board.SCL and board.SDA
accelerometer = adafruit_adxl34x.ADXL345(i2c)
a = 0
time_start = time.time()
while True:
    a = a + 1
    print((time() - time_start) / a)
    print("%f %f %f"%accelerometer.acceleration)
