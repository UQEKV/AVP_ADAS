import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import os
import glob
count = 0
#im_path = "C:\\Users\\rockywin.wang\\Desktop\\01_prj\\20190619_lidar-cam-calib\\01_label_prj\\data\\image_prj\\"
im_path = "../lane_fitting/20200403_shoukai/demo1/fit_result_demo/"
video_path = "../lane_fitting/20200403_shoukai/demo1/"

images = sorted(glob.glob(im_path + '*.png'))
# print(images)
#Edit each frame's appearing time!
fps = 10
count_max = len(images)-112
fourcc = VideoWriter_fourcc(*"mp4v")
#videoWriter = cv2.VideoWriter("3d_box_0927.avi",fourcc,fps,(330,810))
#videoWriter = cv2.VideoWriter("beiqi_demo.avi",fourcc,fps,(240,300))
#videoWriter = cv2.VideoWriter("beiqi_demo.avi",fourcc,fps,(720,2080))

#videoWriter = cv2.VideoWriter("TJP_demo_03.avi",fourcc,fps,(2080,1150))
video_path = video_path + "lineFitting_with_slotLane.mp4"
# videoWriter = cv2.VideoWriter(video_path,fourcc,fps,(1280,720))# w h
videoWriter = cv2.VideoWriter(video_path,fourcc,fps,(480,600))# w h
# cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
#videoWriter = cv2.VideoWriter("test_0920.mp4",fourcc('M', 'J', 'P', 'G'),fps,(330,810))

for im in images:
    #print im
    count = count + 1
    
    frame = cv2.imread(im)
    videoWriter.write(frame)
    if(count > count_max):
        break
	
videoWriter.release()