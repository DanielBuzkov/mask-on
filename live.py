# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2

cap = cv2.VideoCapture(0)

# Create the haar cascade
eyesCascade = cv2.CascadeClassifier("xmls/frontalEyes35x16.xml")
mountCascade = cv2.CascadeClassifier("xmls/Mouth.xml")

while True:
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	eyes = eyesCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=7,
		minSize=(45, 11)
	)

	mouths = mountCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=50,
		minSize=(25, 15)
	)

	# Draw a rectangle around the faces
	for (x1, y1, w1, h1) in eyes:
		has_mouth = False

		if len(mouths) == 0:
			has_mouth = False

		px1 = int(x1 - (0.1 * w1))
		py1 = int(y1 - h1)
		px2 = int(x1 + w1)
		py2 = int(y1 + (2.5 * h1))

		for (x2, y2, w2, h2) in mouths:
			if px1 < x2 and px2 > (x2 + w2) and py1 < y2 and py2 > (y2 + h2):
				has_mouth = True

		if has_mouth:
			cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
		else:
			cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
