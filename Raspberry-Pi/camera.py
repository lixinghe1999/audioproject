import cv2
import time
cap = cv2.VideoCapture(0)

def webcam_save(name, duration):
	t_start = time.time()
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out = cv2.VideoWriter(name + '_' + str(t_start) + '.mp4', fourcc, 20.0, (640, 480))
	count = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		print(count)
		count += 1
		if ret==True:
			out.write(frame)
		if (time.time() - t_start) > duration:
			break
	cv2.destroyAllWindows()
	cap.release()
	out.release()
