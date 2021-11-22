import pyaudio
import wave
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096

def voice_record(name, stream, micframes):
    time_start = time.time()
    frames = []
    for i in range(0, int(micframes / CHUNK)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
    print(micframes / (time.time() - time_start))
    stream.stop_stream()
    stream.close()
    wf = wave.open(name + '_' + str(time_start) + '.wav' , 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    stream.stop_stream()
    stream.close()
    wf.close()
def open_mic_stream(index):
    stream = pyaudio.PyAudio().open(format = FORMAT,
                             channels = CHANNELS,
                             rate = RATE,
                             input = True,
                             input_device_index = index,
                             frames_per_buffer = CHUNK)

    return stream