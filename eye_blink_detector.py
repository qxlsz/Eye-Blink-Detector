from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import webcamvideostream
from webcamvideostream import WebcamVideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import time
import random

start = time.clock()

def eye_aspect_ratio(eye): 
	h1 = dist.euclidean(eye[1], eye[5]) 	#caclulating the euclidean distances b/w vertical eye landmarks 
	h2 = dist.euclidean(eye[2], eye[4])		#caclulating the euclidean distances b/w vertical eye landmarks 
	v1 = dist.euclidean(eye[0], eye[3])		#caclulating the euclidean distances b/w horizontal eye landmarks 
	ear = (h1 + h2) / (2.0 * v1) 			#getting the aspect ratio
	return ear

EYE_AR_THRESH = 0.3  						#eyes aspect ration threshold --> this works for me
EYE_AR_CONSEC_FRAMES = 3 					#number of frames the eye must be below the threshold
COUNTER = 0
TOTAL = 0
Cuteness = 0.0
Cutenessp =""
print("[Qxlsz] : Shape predictor is being loaded . Patience !!! ")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 	#got it from  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 				#get index of left
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] 				#get index of right
print("[Qxlsz] : Magic has to happen to get the perfect feed")
print("[Qxlsz] : By the way you look cute xD")
#vs = FileVideoStream("qxlsz.mp4").start() 		#only if you want to check the blinks from a video lol
#fileStream = True   							#make sure your file is in the same directory and load the video
vs = WebcamVideoStream(src=0).start()
fileStream = False
time.sleep(3.0)
while True:
	if fileStream and not vs.more(): #if file , check any frame if leftover in buffer
		break
	frame = vs.read() 
	frame = imutils.resize(frame, width=880) #resize , i kept 880 as that's the interstate i live . 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0) #magic happens here where it detects face
	for rect in rects:
		Cuteness = random.uniform(96.4, 99.7)
		Cutenessp = (str(float(Cuteness)) + "%")
		shape = predictor(gray, rect) #for all faces in the predictor 
		shape = face_utils.shape_to_np(shape)
		#extract both eyes coordinates and compute aspect ratio
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			# reset the eye frame counter
			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Cuteness: {:}".format(Cutenessp), (600, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)		
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()