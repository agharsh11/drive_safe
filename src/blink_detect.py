from scipy.spatial import distance
import dlib
import cv2
import numpy as np
from timeit import default_timer as timer

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	aspect = (A + B) / (2.0 * C)
	return aspect
	
def shape_to_np(shape, dtype="int"):
	k = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		k[i] = (shape.part(i).x, shape.part(i).y)
	return k
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
cap=cv2.VideoCapture(0)
flag=0
blink_rate=30
blink=0
last=timer()
time=0
while True:
	start = timer()
	time=time+start-last;
	last=start
	if time>5:
		if blink>15:
			print "strain" 
		blink=0
		time=0
	#print time
	ret, frame=cap.read()
	frame = cv2.resize(frame, (450,300))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		#print type(shape)
		shape = shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftaspect = eye_aspect_ratio(leftEye)
		rightaspect = eye_aspect_ratio(rightEye)
		aspect = (leftaspect + rightaspect) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if aspect < thresh:
			flag += 1
			blink = blink + 1
			print (blink)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("sleeping")
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()