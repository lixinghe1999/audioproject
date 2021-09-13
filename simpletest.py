import time
import board
import busio
import adafruit_adxl34x

i2c = busio.I2C(board.SCL, board.SDA)
accelerometer = adafruit_adxl34x.ADXL345(i2c)
a = 0
time_start = time.time()
while True:
    a = a + 1
    print((time() - time_start) / a)
    print("%f %f %f"%accelerometer.acceleration)
