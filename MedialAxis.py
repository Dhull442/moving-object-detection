import numpy as np;
import cv2
import os, sys
import math

kernel = np.ones((5,5),np.uint8);
cutoff = 0;
prev_x1 = prev_x2 = prev_y1 = prev_y2 = 0
last5_slopes = []
epsilon = 3;
persist = False;

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
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	ret = 1
	out = cv2.VideoWriter('result_'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
	while(ret or cap.isOpened()):
		ret, frame = cap.read()
		if(ret != True):
			break
		fgmask = fgbg.apply(frame)
		opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		cv2.imwrite('temp.jpg',opening)
		# cv2.imshow('open',opening)
		labeled = applySobel('temp.jpg',frame)
		# cv2.imshow('frame',labeled)
		# cv2.
		if(ret):
			out.write(frame)
		k = cv2.waitKey(30) & 0xff
		if k==27:
			break;
	cap.release()
	out.release()
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
	#edges = cv2.Canny(gray,50,150,apertureSize = 3) #23,55
	edges = cv2.Canny(gray,23,60,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/720,120) #150 #90
	if lines is not None:
		# number = 0;
		# x1_mean = y1_mean = x2_mean = y2_mean = 0;
		x1_list = [];
		x2_list = [];
		y1_list = [];
		y2_list = [];
		for line in lines:
			rho,theta = line[0]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			if (y1 <= y2):
				x1_list.append(x1)
				x2_list.append(x2)
				y1_list.append(y1)
				y2_list.append(y2)
			else :
				x1_list.append(x2)
				x2_list.append(x1)
				y1_list.append(y2)
				y2_list.append(y1)
			# number = number + 1;
			# x1_mean = (x1_mean)+(x1-x1_mean)/number
			# y1_mean = (y1_mean)+(y1-y1_mean)/number
			# x2_mean = (x2_mean)+(x2-x2_mean)/number
			# y2_mean = (y2_mean)+(y2-y2_mean)/number
		x1_median = int(np.ma.median(x1_list))
		x2_median = int(np.ma.median(x2_list))
		y1_median = int(np.ma.median(y1_list))
		y2_median = int(np.ma.median(y2_list))
		# cv2.line(original,(int(x1_mean),int(y1_mean)),(int(x2_mean),int(y2_mean)),(0,0,255),2)

		lines = cv2.HoughLinesP(edges,0.5,np.pi/720,75,70,8)
		l = 0;
		xmin = ymin = 100000;
		xmax = ymax = -100000;
		if lines is not None:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				if(y1<ymin): ymin = y1;
				if(x1<xmin): xmin = x1;
				if(x2>xmax): xmax = x2;
				if(y2>ymax): ymax = y2;
				if(y2<ymin): ymin = y2;
				if(x2<xmin): xmin = x2;
				if(x1>xmax): xmax = x1;
				if(y1>ymax): ymax = y1;
				# cv2.line(original,(x1,y1),(x2,y2),(0,0,255),2)
			# l = math.sqrt((xmin-xmax)**2 + (ymin-ymax)**2)
		# cv2.line(original,(int(x1_mean)+int(x2_mean)-xmin,int(y2_mean)+int(y1_mean)-ymin),(int(xmin),ymin),(0,0,255),2)
		# slope_mean = abs((y2_mean - y1_mean)/(x2_mean-x1_mean))
		# slope_median = abs((y2_median - y1_median)/(x2_median-x1_median))
		# print(slope_mean-slope_median);
		# cv2.line(original,(int(x1_mean),int(y1_mean)),(int(x2_mean),int(y2_mean)),(0,0,255),2)
		global cutoff, prev_x1, prev_x2, prev_y1, prev_y2, last5_slopes
		if(x2_median - x1_median is not 0):
			slope = (y2_median - y1_median)/(x2_median - x1_median)
		else:
			slope = last5_slopes[len(last5_slopes)-1]

		prev_slope = slope
		use_this = True;

		all_neg = True;
		all_pos = True;
		for x in last5_slopes:
			if (x > 0):
				all_neg = False;
			elif (x < 0):
				all_pos = False;

		if (len(last5_slopes) > 4) and ((all_pos and slope < 0 and last5_slopes[len(last5_slopes)-1] < 200 ) or (all_neg and slope > 0 and last5_slopes[len(last5_slopes)-1] > -200)):
			use_this = False;
		else:
			last5_slopes.append(slope)
			while (len(last5_slopes) > 5):
				last5_slopes.pop(0)

		# print(str(prev_slope) + " " + str(slope));

		# if(prev_slope != 0):
		# 	if(slope - prev_slope > epsilon or slope -prev_slope < epsilon):
		# 		slope = prev_slope;
		x_lim_up = (0 - y1_median)/slope + x1_median
		if (cutoff == 0):
			cutoff = ymax;
		elif (cutoff > 0 and ymax > 0):
			cutoff = 0.9*cutoff + 0.1*ymax;
		# print(cutoff)
		x_lim_down = (cutoff - y1_median)/slope + x1_median
		# print(str(x1_median) + " " + str(y1_median) + " " + str(x2_median) + " " + str(y2_median))
		# cv2.line(original,(int(x1_median),int(y1_median)),(int(x2_median),int(y2_median)),(0,255,255),3)
		if use_this:
			cv2.line(original,(int(x_lim_up),int(0)),(int(x_lim_down),int(cutoff)),(0,0,255),3)
		else:
			cv2.line(original,(int(prev_x1),int(prev_y1)),(int(prev_x2),int(prev_y2)),(0,0,255),3)
		prev_x1 = x_lim_up;
		prev_y1 = 0;
		prev_x2 = x_lim_down;
		prev_y2 = cutoff;
		# cv2.imshow('asd',img)
	elif persist:
		cv2.line(original,(int(prev_x1),int(prev_y1)),(int(prev_x2),int(prev_y2)),(0,0,255),3)
		# prev_slope = 0;
	return original


# with open('filelist.txt') as f:
	# lines = f.readlines();
	# for file in lines:
	# 	name=lines[i].split('.',1)[0];		#5,8,10
	# 	cutoff = 0;
	# 	BGSubtraction(name);
	#FrameCapture(name);
BGSubtraction("1")
BGSubtraction("2")
BGSubtraction("3")
BGSubtraction("4")
BGSubtraction("5")
BGSubtraction("6")
BGSubtraction("7")
BGSubtraction("8")
BGSubtraction("9")
BGSubtraction("10")
