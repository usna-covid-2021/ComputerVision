#DetectFace.py
#   This program will detect faces and a second feature
#   using live video from the default webcam.

# USAGE
# python DetectFace.py 

# import the necessary packages (specifically the opencv package)
import imutils
import cv2

# create a video object for the default webcam
camera = cv2.VideoCapture(0)
 
# create cascade classifier which uses a face database
face_cascade = cv2.CascadeClassifier('HarrXML\haarcascade_frontalface_alt2.xml')
second_cascade = cv2.CascadeClassifier('HarrXML\haarcascade_eye.xml')

print("Press ESC or q to end program")
frame_number=0; # this variable is used to count the frames
# keep looping
while True:
	frame_number=frame_number+1 # count each frame that is processed
	
	# grab the current frame
	(grabbed, img) = camera.read()

	# resize the frame, optional 
	#img = cv2.resize(img, (0,0), fx=2.0, fy=2.0)

	# convert color image to gray scale image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# find faces in entire image (uncomment one)
	faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)
	#faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(200, 200),maxSize=(400, 400))

	# draw retangle around each detected face
	for (x,y,w,h) in faces:
		#print "Face: Left=%d Top=%d Width=%d Height=%d" % (x, y, w, h)

		# draw retangle around this detected face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

		# crop image to this detected face
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		
		# find second feature in this cropped image
		second = second_cascade.detectMultiScale(roi_gray,minNeighbors=10)
		
		# draw retangle around each detected feature
		for (sx,sy,sw,sh) in second:
			cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
			#print "	  Second: Left=%d Top=%d Width=%d Height=%d" % (sx, sy, sw, sh)
			
	# print frame number on image
	cv2.putText(img, str(frame_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 2)
 
    # display entire image which contains rectangles drawn on it 
	cv2.imshow("Frame", img)
	
	
	# check keyboard for a keypress
	key = cv2.waitKey(1) & 0xFF

	# if the ESC or 'q' key is pressed, stop the loop
	if key == 27 or key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
