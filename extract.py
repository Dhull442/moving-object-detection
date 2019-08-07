import cv2
import os

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


with open('filelist.txt') as f:
    lines = f.readlines();
    os.mkdir("dataset/images")
    for line in lines:
        name=line.split('.',1)[0]
        FrameCapture(name);
    # FrameCapture("")
