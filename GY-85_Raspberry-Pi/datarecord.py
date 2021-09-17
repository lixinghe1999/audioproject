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

def voice_record(name, stream, frames):
    block = stream.read(frames, exception_on_overflow=False)
    stream.stop_stream()
    stream.close()
    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(block)
    wf.close()
    print(frames/(time.time()-time_start))

def open_mic_stream(index, frames):
    stream = pyaudio.PyAudio().open(format = FORMAT,
                             channels = CHANNELS,
                             rate = RATE,
                             input = True,
                             input_device_index = index,
                             frames_per_buffer = frames)

    return stream
def acc_save(time_start):
    a = 0
    adxl345 = i2c_adxl345(1)
    adxl345.setdatarate(0x0F)
    accwriter = open('acc.txt', 'w')
    acc = ''
    while (a < 10000):
        if adxl345.getInterruptStatus():
            a = a + 1
            (x1, y1, z1) = adxl345.getAxes()
            acc = acc + str(x1) + ' ' + str(y1) + ' ' + str(z1) + ' ' + str(time.time()) + '\n'
        else:
            accwriter.write(acc)
            acc = ''
    print('acc 10000')
    print(a/(time.time() - time_start))
def gyro_save(time_start):
    b = 0
    itg3205 = i2c_itg3205(1)
    gyrowriter = open('gyro.txt', 'w')
    while (b < 10000):
        itgready, dataready = itg3205.getInterruptStatus()
        if dataready:
            (x2, y2, z2) = itg3205.getDegPerSecAxes()
            gyrowriter.write(str(x2) + ' ' + str(y2) + ' ' + str(z2) + ' ' + str(time.time()) + '\n')
            b = b + 1
    print('gyro 10000')
    print(b/(time.time() - time_start))
def compass_save(time_start):
    c = 0
    hmc5883l = i2c_hmc5883l(1)
    hmc5883l.setContinuousMode()
    hmc5883l.setDeclination(3, 5)
    compasswriter = open('compass.txt', 'w')
    while (c<10):
        (x3, y3, z3) = hmc5883l.getAxes()
        compasswriter.write(str(x3) + ' ' + str(y3) + ' ' + str(z3) + ' ' + str(time.time()) + '\n')
        c = c + 1
        time.sleep(0.5)
if __name__ == "__main__":

    time_start = time.time()
    # thread1 = threading.Thread(target = acc_save, args = (time_start,))
    # thread2 = threading.Thread(target = gyro_save, args =(time_start,))
    # thread3 = threading.Thread(target = voice_record, args=('mic1.wav', open_mic_stream(1, 100000), 100000))
    # thread4 = threading.Thread(target = voice_record, args=('mic2.wav', open_mic_stream(2, 100000), 100000))
    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()
    # thread1.join()
    # thread2.join()
    # thread3.join()
    # thread4.join()
    # #
    thread1 = Process(target = acc_save, args=(time_start,))
    thread2 = Process(target = compass_save(), args=(time_start,))
    #thread2 = Process(target = gyro_save, args =(time_start,))
    thread3 = Process(target = voice_record, args=('mic1.wav', open_mic_stream(1, 1024), 120000))
    thread4 = Process(target = voice_record, args=('mic2.wav', open_mic_stream(2, 1024), 120000))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

