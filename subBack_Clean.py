import numpy as np;
import cv2
import os, sys

kernel = np.ones((8,8),np.uint8)

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
	fgbg = cv2.createBackgroundSubtractorMOG2(200,16,False);
	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		cv2.imwrite('temp.jpg',opening)
		sobeled = applySobel('temp.jpg')
		cv2.imshow('frame',sobeled)
		k = cv2.waitKey(30) & 0xff
		if k==27:
			break;
			
	cap.release()
	cv2.destroyAllWindows()
	
def applySobel(image_path):
	scale = 1
	delta = 0
	ddepth = cv2.CV_16S	
	src = cv2.imread(image_path, cv2.IMREAD_COLOR)
	src = cv2.GaussianBlur(src, (3, 3), 0)
	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	# Gradient-Y
	grad_y = cv2.Scharr(gray,ddepth,0,1)
	#grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)  
	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
	cv2.imwrite('temp.jpg',grad)
	houghed = applyHough('temp.jpg')
	return houghed
	#cv.imshow(window_name, grad)
	#cv.waitKey(0)
	
def applyHough(image):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/180,120)
	if lines is not None:
		number = 0;
		x1_mean = y1_mean = x2_mean = y2_mean = 0;
		
		for line in lines:
			rho,theta = line[0]
			if rho < 0:
				rho*=-1
				theta-=np.pi
			number = number + 1;
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			x1_mean = (x1_mean+x1)/number
			y1_mean = (y1_mean+y1)/number
			x2_mean = (x2_mean+x2)/number
			y2_mean = (y2_mean+y2)/number
			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
		x1 = round(x1_mean);
		x2 = round(x2_mean);
		y1 = round(y1_mean);
		y2 = round(y2_mean);
	
	return img


with open('filelist.txt') as f:
	lines = f.readlines();
	name=lines[1].split('.',1)[0];
	BGSubtraction(name);
	#FrameCapture(name);
