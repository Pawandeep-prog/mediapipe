import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

facmesh = mp.solutions.face_mesh
face = facmesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

while True:

	_, frm = cap.read()
	print(frm.shape)
	break
	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = face.process(rgb)
	if op.multi_face_landmarks:
		for i in op.multi_face_landmarks:
			print(i.landmark[0].y*480)
			draw.draw_landmarks(frm, i, facmesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 255), circle_radius=1))


	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
