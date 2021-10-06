import time
from BMI160_i2c import Driver

sensor = Driver(0x68) # change address if needed
sensor.set_accel_rate(12)
time_start = time.time()
num = 10000
a = 0
accwriter = open('bmi_acc.txt', 'w')
acc = ''
time_start = time.time()
while (a < num):
    if sensor.getIntDataReadyStatus():
        a = a + 1
        data = sensor.getAcceleration()
        acc = acc + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(time.time()) + '\n'
    else:
        accwriter.write(acc)
        acc = ''
print(num/(time.time() - time_start))
