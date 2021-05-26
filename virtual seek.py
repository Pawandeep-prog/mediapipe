import cv2
import mediapipe as mp 
import numpy as np
import time
import autopy
import pyautogui

x,y=0,0
cx,cy = 0,0

startx, starty, endx, endy = 0,0,0,0
k = 0

def onMouse(event, x, y, flags, param):
	global startx, starty, endx, endy, k

	if event == cv2.EVENT_LBUTTONDOWN:
		startx, starty = x*2,y*2
		k += 1
		print("set s")

	elif event == cv2.EVENT_RBUTTONDOWN:
		endx, endy = x*2, y*2
		k += 1
		print("set e")

cv2.namedWindow("setpoint")
cv2.setMouseCallback("setpoint", onMouse)

print('going to take screenshot! /n show only the player window clearly !')
time.sleep(2)
im = pyautogui.screenshot().convert('RGB')
im = np.array(im)
im = im[:, :, ::-1]

im = cv2.resize(im, (960,540))

while True:
	cv2.imshow("setpoint", im)
	if cv2.waitKey(1) == 27 or k==2:
		cv2.destroyAllWindows()
		break


hnds = mp.solutions.hands
hnds_mesh = hnds.Hands(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mp.solutions.drawing_utils

cap  = cv2.VideoCapture(0)

done = False
pset = False

linewidth = (endx - startx)
seekwidth = 200
mul = int(linewidth/seekwidth)
ptime, ctime = 0,0
rad = 30
while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hnds_mesh.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			point8 = i.landmark[8]
			point5 = i.landmark[5]
			draw.draw_landmarks(frm, i, hnds.HAND_CONNECTIONS)

			if int(abs(point8.y*480 - point5.y*480)) > 35:
				# one
				if not(pset):
					pset = True
					ptime = time.time()

				ctime = time.time()

				if (ctime - ptime) > 1:
					if not(done):
						x = int(point8.x*640)
						y = int(point8.y*480)
						done = True

					cx = int(point8.x*640)

					if cx < x:
						cx = x
					elif cx > (x+seekwidth):
						cx = x+seekwidth

					cv2.line(frm, (x,y), (x+seekwidth, y), (255,0,255), 6)
					cv2.circle(frm, (cx,y), 9, (0,255,0), -1)

					autopy.mouse.move((cx-x)*mul+startx,starty)

				else:
					cv2.circle(frm, (int(point8.x*640),int(point8.y*480)), rad, (0,255,255), 3)
					if rad > 4:
						rad -= 1

			else:
				# not one
				if done:
					autopy.mouse.click()
				done = False
				pset = False
				ctime = 0
				ptime = 0
				rad = 30

	cv2.imshow("window", frm)


	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break