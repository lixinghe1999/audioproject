import pyaudio

def getaudiodevices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i)
        print(p.get_device_info_by_index(i))

getaudiodevices()