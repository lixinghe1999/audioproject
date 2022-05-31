import cv2
import time
cap = cv2.VideoCapture(0)

def webcam_save(name, num):
	t_start = time.time()
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out = cv2.VideoWriter(name + '_' + str(t_start) + '.mp4', fourcc, 20.0, (640, 480))
	count = 0
	for i in range(num):
		ret, frame = cap.read()
		print(count)
		out.write(frame)
	cv2.destroyAllWindows()
	cap.release()
	out.release()
