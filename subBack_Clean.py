import numpy as np
import cv2
import os

erosiokernel = np.ones((4,4),np.uint8)
openingkernel = np.ones((5,5),np.uint8)
def hough(gray,orig):
	# img = cv2.imread('dave.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)

	lines = cv2.HoughLines(edges,1,np.pi/180,200)
	if(lines is not None):
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(orig,(x1,y1),(x2,y2),(0,0,255),2)
	cv2.imshow('houghlines',orig)
def sobelfilter(image,orig):
	scale = 1
	delta = 0
	ddepth = cv2.CV_16S
	grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
	grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)
	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
	cv2.imshow('sobel', grad)
	# h = cv2.HoughLines(grad)
	hough(grad,orig)

	# cv.waitKey(0)
	return

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
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	ret = 1;
	while(ret & cap.isOpened()):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		cv2.imshow('main',fgmask)
		# erosion = cv2.erode(fgmask,erosiokernel,iterations=1)
		# opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, openingkernel)
		# cv2.imshow('frame',opening)
		sobelfilter(fgmask,frame)
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
