
from bmi160 import bmi160_accsave, bmi160_gyrosave
from gy85 import gy85_gyrosave, gy85_compasssave, gy85_accsave
from mic import open_mic_stream, voice_record


from multiprocessing import Process
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--time', action = "store",type=int, default = 5, required=False, help='time of data recording')
    parser.add_argument('--compass', action="store", type=int, default = 0, required=False, help='use compass')
    parser.add_argument('--acctype', action="store", type=int, default=0, required=False, help='gy or bmi')
    parser.add_argument('--port', action="store", type=int, default=1, required=False, help='port/ i2c bus')
    args = parser.parse_args()

    gyaccframe = args.time * 3000
    bmiaccframe = args.time * 1600
    port = args.port
    gyroframe = args.time * 1500
    compassframe = args.time * 15
    micframe = args.time * 44100
    if args.acctype == 0:
        thread1 = Process(target=bmi160_accsave, args=('bmiacc', bmiaccframe, port))
        thread4 = Process(target=bmi160_gyrosave, args=('bmigryo', bmiaccframe, port))
        thread2 = Process(target=voice_record, args=('mic1', open_mic_stream(1, micframe), micframe))
        thread3 = Process(target=voice_record, args=('mic2', open_mic_stream(2, micframe), micframe))
        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()
    elif args.acctype == 1:
        thread1 = Process(target=gy85_accsave, args=('gyacc', gyaccframe, port))
        thread2 = Process(target=voice_record, args=('mic1', open_mic_stream(1), micframe))
        thread3 = Process(target=voice_record, args=('mic2', open_mic_stream(2), micframe))
        thread1.start()
        thread2.start()
        thread3.start()
        thread1.join()
        thread2.join()
        thread3.join()
    else:
        thread1 = Process(target=bmi160_accsave, args=('bmiacc', bmiaccframe, port))
        thread1.start()
        thread1.join()
    # if args.compass == 1:
    #     thread4 = Process(target=gy85_compasssave, args=(compassframe,))

