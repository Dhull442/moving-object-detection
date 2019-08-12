import numpy as np;
import cv2
import os

kernel = np.ones((5,5),np.uint8)

# Function to extract frames
def FrameCapture(name):
	print("working "+name)
	os.mkdir("dataset/images/"+name)
	cap = cv2.VideoCapture("dataset/"+name+".mp4")
	count = 0
	ret = 1
	while (ret & cap.isOpened()):
		ret, image = cap.read();#cv2.imshow('frame'+str(count), image);
		cv2.imwrite("dataset/images/"+name+"/f" + str(count)+".jpg", image)
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break
		print("frame "+str(count)+" written");
		count += 1
	cap.release()
	cv2.destroyAllWindows()
	
def BGSubtraction(name):
	cap = cv2.VideoCapture("dataset/"+name+".mp4")
	fgbg = cv2.BackgroundSubtractorMOG2(200,40,False)
	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		cv2.imshow('frame',opening)
		k = cv2.waitKey(30) & 0xff
		if k==27:
			break;
			
	cap.release()
	cv2.destroyAllWindows()


with open('filelist.txt') as f:
	lines = f.readlines();
	name=lines[1].split('.',1)[0];
	BGSubtraction(name);
	#FrameCapture(name);
