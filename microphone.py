
import pyaudio
import threading
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INPUT_BLOCK_TIME = 0.0001
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

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
    return 0

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
    thread1 = threading.Thread(target=voice_record, args=('mic1.wav', open_mic_stream(1), 100000))
    thread2 = threading.Thread(target=voice_record, args=('mic2.wav', open_mic_stream(2), 100000))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()