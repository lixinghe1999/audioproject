
import pyaudio
import threading

INITIAL_TAP_THRESHOLD = 200
INITIAL_TAP_THRESHOLD1 = 200
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)
CHANNELS = 1
RATE = 44100
INPUT_BLOCK_TIME = 0.0001
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
def voice_record(stream, frames):
    block = stream.read(frames, exception_on_overflow=False)
    print(type(block), len(block))
    return block

def open_mic_stream(index, rate = 44100):
    stream = pyaudio.PyAudio().open(format = FORMAT,
                             channels = CHANNELS,
                             rate = RATE,
                             input = True,
                             input_device_index = index,
                             frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

    return stream

if __name__ == "__main__":
    pa = pyaudio.PyAudio()
    thread1 = threading.Thread(target=voice_record, args=(1,))
    thread2 = threading.Thread(target=voice_record, args=(2,))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()