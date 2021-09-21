import sys
import threading
sys.path.insert(1,'/home/pi/audioproject/GY-85_Raspberry-Pi/i2clibraries')
from i2c_adxl345 import *
from i2c_itg3205 import *
from i2c_hmc5883l import *
import time
import pyaudio
import threading
import wave
from multiprocessing import Process
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
#CHUNK = 8192

def voice_record(name, stream, frames):
    time_start = time.time()
    block = stream.read(frames, exception_on_overflow=False)
    print(frames / (time.time() - time_start))
    wf = wave.open(name + '_' + str(time_start) + '.wav' , 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(block)
    stream.stop_stream()
    stream.close()
    wf.close()
def open_mic_stream(index, frames):
    stream = pyaudio.PyAudio().open(format = FORMAT,
                             channels = CHANNELS,
                             rate = RATE,
                             input = True,
                             input_device_index = index,
                             frames_per_buffer = frames)

    return stream
def acc_save(num):
    a = 0
    adxl345 = i2c_adxl345(1)
    adxl345.setdatarate(0x0F)
    accwriter = open('acc.txt', 'w')
    acc = ''
    time_start = time.time()
    while (a < num):
        if adxl345.getInterruptStatus():
            a = a + 1
            (x1, y1, z1) = adxl345.getAxes()
            acc = acc + str(x1) + ' ' + str(y1) + ' ' + str(z1) + ' ' + str(time.time()) + '\n'
        else:
            accwriter.write(acc)
            acc = ''
    print('acc', a / (time.time() - time_start))

def gyro_save(num):
    b = 0
    itg3205 = i2c_itg3205(1)
    gyrowriter = open('gyro.txt', 'w')
    time_start = time.time()
    while (b < num):
        itgready, dataready = itg3205.getInterruptStatus()
        if dataready:
            (x2, y2, z2) = itg3205.getDegPerSecAxes()
            gyrowriter.write(str(x2) + ' ' + str(y2) + ' ' + str(z2) + ' ' + str(time.time()) + '\n')
            b = b + 1
    print('gyro', b/(time.time() - time_start))
def compass_save(num):
    c = 0
    hmc5883l = i2c_hmc5883l(1)
    hmc5883l.setContinuousMode()
    hmc5883l.setDeclination(3, 5)
    compasswriter = open('compass.txt', 'w')
    num / 3000
    while (c < int(num / 300)):
        (x3, y3, z3) = hmc5883l.getAxes()
        compasswriter.write(str(x3) + ' ' + str(y3) + ' ' + str(z3) + ' ' + str(time.time()) + '\n')
        c = c + 1
        time.sleep(0.1)
if __name__ == "__main__":
    num = 10000
    micframe = num * 10
    thread1 = Process(target = acc_save, args=(num,))
    thread2 = Process(target = compass_save, args=(num, ))
    #thread2 = Process(target = gyro_save, args =(num,))
    thread3 = Process(target = voice_record, args=('mic1', open_mic_stream(1, micframe), micframe))
    thread4 = Process(target = voice_record, args=('mic2', open_mic_stream(2, micframe), micframe))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

