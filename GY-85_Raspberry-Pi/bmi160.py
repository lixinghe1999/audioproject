
from BMI160_i2c import Driver

print('Trying to initialize the sensor...')
sensor = Driver(0x68) # change address if needed
print('Initialization done')
sensor.set_accel_rate(13)
print(sensor.get_accel_rate())
while True:
  data = sensor.getAcceleration()
  # fetch all gyro and acclerometer values
  print({
    'ax': data[0],
    'ay': data[1],
    'az': data[2]
  })