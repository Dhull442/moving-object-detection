import numpy as np;
import cv2
import math
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
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG();
	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		cv2.imwrite('temp.jpg',opening)
		sobeled = applySobel('temp.jpg',frame)
		cv2.imshow('frame',sobeled)
		k = cv2.waitKey(30) & 0xff
		if k==27:
			break;

	cap.release()
	cv2.destroyAllWindows()

def applySobel(image_path,original):
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
	houghed = applyHough('temp.jpg',original)
	return houghed
	#cv.imshow(window_name, grad)
	#cv.waitKey(0)

def applyHough(image,original):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	lines = cv2.HoughLinesP(edges,0.5,np.pi/720,100,25,10)
	xmin = ymin = 100000;
	xmax = ymax = 0;
	if lines is not None:
		number = 0;
		for line in lines:
			x1,y1,x2,y2 = line[0]
			number = number + 1;
			if(y1<ymin): ymin = y1;
			if(x1<xmin): xmin = x1;
			if(x2>xmax): xmax = x2;
			if(y2>ymax): ymax = y2;
			# cv2.line(original,(x1,y1),(x2,y2),(0,0,255),2)
		# x_mid = (x1_mean + x2_mean) / 2;
		# y_mid = (y1_mean + y2_mean) / 2;
		# m = math.tan(a_avg);
		# print(m);
		# # print(x_mid)
		# c = x_mid + (y_mid/m);
		# print(c)
		# cv2.line(original,(int(c),0),(int(2*x_mid-c),int(2*y_mid)),(255,0,255),4)
	l = math.sqrt((xmin-xmax)**2 + (ymin-ymax)**2)
	lines = cv2.HoughLines(edges,0.5,np.pi/720,100)
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
			x1 = int(x0 + l*(-b))
			y1 = int(y0 + l*(a))
			x2 = int(x0 - l*(-b))
			y2 = int(y0 - l*(a))
			# a=np.arctan(y2-y1/x2-x1)
			# a_avg = a_avg + (a-a_avg)/number
			x1_mean = (x1_mean)+(x1-x1_mean)/number
			y1_mean = (y1_mean)+(y1-y1_mean)/number
			x2_mean = (x2_mean)+(x2-x2_mean)/number
			y2_mean = (y2_mean)+(y2-y2_mean)/number
			# if(y1<ymin): ymin = y1;
			# if(y2<ymin): ymin = y2;
			# if(x1<xmin): xmin = x1;
			# if(x2<xmin): xmin = x2;
			# cv2.line(original,(x1,y1),(x2,y2),(0,0,255),2)
		cv2.line(original,(int(x1_mean),int(y1_mean)),(int(x2_mean),int(y2_mean)),(0,255,255),3)
	return original


with open('filelist.txt') as f:
	lines = f.readlines();
	name=lines[1].split('.',1)[0];
	BGSubtraction(name);
	#FrameCapture(name);
