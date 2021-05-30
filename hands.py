import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils
frm = cv2.imread("hands.jpg")
while True:
	#_, frm = cap.read()
	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hands_mesh.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS, 
				landmark_drawing_spec=draw.DrawingSpec(color = (255, 0,0),circle_radius=4, thickness=3),
				connection_drawing_spec=draw.DrawingSpec(thickness=3, color=(0,0,255)))


	cv2.imshow("window", frm)
	cv2.imwrite("hand.jpg", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break