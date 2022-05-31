import cv2
import time
cap = cv2.VideoCapture(0)

def webcam_save(name, duration):
	# a = 0
	# cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
	# while(a < num):
	# 	a += 1
	# 	ret, image = cam.read()
	# 	time.sleep(0.04)
	# 	cv2.imwrite(name + '_' + str(time.time()) + '.jpg', image)
	t_start = time.time()
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out = cv2.VideoWriter(name + '_' + str(t_start) + '.mp4', fourcc, 20.0, (640, 480))
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret==True:
			out.write(frame)
		if (time.time() - t_start) > duration:
			break
	cv2.destroyAllWindows()
	cap.release()
	out.release()
