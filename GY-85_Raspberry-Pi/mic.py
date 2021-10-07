import pyaudio
import wave
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

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