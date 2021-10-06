import time
from BMI160_i2c import Driver

sensor = Driver(0x68) # change address if needed
sensor.set_accel_rate(13)
time_start = time.time()
i = 0
while(i<100):
    if sensor.getIntDataReadyStatus():
        data = sensor.getAcceleration()
        i = i + 1
print(100/(time.time() - time_start))
#while True:
  #data = sensor.getAcceleration()
  # fetch all gyro and acclerometer values
  # print({
  #   'ax': data[0],
  #   'ay': data[1],
  #   'az': data[2]
  # })