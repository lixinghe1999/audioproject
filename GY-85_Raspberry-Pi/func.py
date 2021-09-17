import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from imuplot import read_data, interpolation
import matplotlib.pyplot as plt
def euler(accel, compass):
    accelX, accelY, accelZ = accel
    compassX, compassY, compassZ = compass
    pitch = 180 * math.atan2(accelX, math.sqrt(accelY*accelY + accelZ*accelZ))/math.pi
    roll = 180 * math.atan2(accelY, math.sqrt(accelX*accelX + accelZ*accelZ))/math.pi
    mag_x = compassX * math.cos(pitch) + compassY*math.sin(roll)*math.sin(pitch) + compassZ*math.cos(roll)*math.sin(pitch)
    mag_y = compassY * math.cos(roll) - compassZ * math.sin(roll)
    yaw = 180 * math.atan2(-mag_y, mag_x)/math.pi
    return [roll, pitch, yaw]
def re_coordinate(acc_data, compass):
    compass_num = 0
    rotationmatrix = np.eye(3)
    for i in range(np.shape(acc_data)[0]):
        data1 = acc_data[i, :]
        data2 = compass[compass_num, :]
        if data1[-1] > data2[-1]:
            compass_num = compass_num + 1
            print(euler(data1[:-1], data2[:-1]))
            rotationmatrix = R.from_euler('xyz', euler(data1[:-1], data2[:-1]), degrees=True).as_matrix()
        acc_data[i, :-1] = np.dot(rotationmatrix, data1[:-1].transpose())
    return acc_data
acc_data = read_data('../acc.txt')
compass = read_data('../compass.txt')
acc_data = re_coordinate(acc_data, compass)
acc_data = interpolation(acc_data)

fig, axs = plt.subplots(3, 1)
for i in range(3):
    axs[i].plot(acc_data[:, i])
plt.show()