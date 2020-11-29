"""
util tool function for relationship
"""
import numpy as np
import pandas as pd 
import matplotlib as plt

import os
import os.path

import tkinter as tk 
from PIL import Image,ImageTk
import argparse

import cv2

#frame 
FRAME_WIDTH = 320
FRAME_HIGHT = 192

#object cls
FRONT_WHEEL_CLS = 7
REAR_WHEEL_CLS = 6
CAR_CLS = 1




"""
show the checked img and origin img in a window.
"""

class CompareShow(object):
    def __init__(self,ori_img_path,matched_img_path):
        self.img_l_path = ori_img_path
        self.img_r_path = matched_img_path
        self.matched_result = True

    
    def show(self):
        #create a window for compare the img.
        window = tk.Tk()
        window.title('conform window')
        window.geometry('700x250')


        img_open_left = Image.open(self.img_l_path)
        img_open_right = Image.open(self.img_r_path)

        #img_left : origin img
        #img_right: matched img
        img_left = ImageTk.PhotoImage(img_open_left)
        img_right = ImageTk.PhotoImage(img_open_right)


        img_l = tk.Label(window)
        img_l.configure(image=img_left)
        img_l.image=img_left

        img_r = tk.Label(window)
        img_r.configure(image=img_right)
        img_r.image=img_right
        img_r.config(bg='red')

        img_l.place(x=15,y=8,anchor='nw')
        img_r.place(x=365,y=8,anchor='nw')

        var1 = tk.IntVar()  # 定义var1和var2整型变量用来存放选择行为返回值
        var2 = tk.IntVar()
        c1 = tk.Checkbutton(window, text='Wrong',variable=var1, onvalue=1, offvalue=0, command=lambda: self.select_left(window,var1))    # 传值原理类似于radiobutton部件
        c1.place(x=320,y=225,anchor='nw')

        c2 = tk.Checkbutton(window, text='Right',variable=var2, onvalue=1, offvalue=0, command=lambda: self.select_right(window,var2))
        c2.place(x=370,y=225,anchor='nw')

        window.mainloop()


    # select left represent the match is wrong in the frame.
    def select_left(self,window,var):
        if var.get() == 1:
            window.destroy()
            self.matched_result = False


    # select left represent the match is right in the frame.
    def select_right(self,window,var):
        if var.get() == 1:
            window.destroy()
            self.matched_result = True


    def get_matched_result(self):
        return self.matched_result



"""
base func for process csv data
"""
# frame_title=['frameid','channel','cls', \
#             'cb_x1','cb_y1','cb_x2''cb_y2', \
#             'wb_x1','wb_y1','wb_x2','wb_y2',\
#             'iou','cb_index','wb_index','carWheelCls','mode']
# carPos_title = ['x1','y1','x2','y2']
# carWheelPos_title = ['x1','y1','x2','y2']

def gen_frames_from_csv_data(data):
    if data is None:
        return None

    title = ['frameid','channel','cls', \
             'cb_x1','cb_y1','cb_x2','cb_y2', \
             'wb_x1','wb_y1','wb_x2','wb_y2',\
             'iou','cb_index','wb_index','carWheelCls','mode']

    
    frames = []
    i = 0
    while i <len(data):
        frame = pd.DataFrame(columns=title)
        frame = frame.append(data.iloc[i])
        k = 1
        while i+k < len(data):
            if (data.iloc[i+k].loc['frameid'],data.iloc[i+k].loc['channel']) == (data.iloc[i].loc['frameid'],data.iloc[i].loc['channel']):
                frame = frame.append(data.iloc[i+k])
                k += 1
            else:
                break
        i += k
        frames.append(frame)

    return frames



def get_imgname_list(file_path):
    imgs_name_list=[]
    imgs = os.listdir(file_path)
    for img in imgs:
        img_name,img_ext = os.path.splitext(img)
        img_name_list = img_name.split('_')
        imgs_name_list.append(img_name_list)
    return imgs_name_list



#get image path from a file by info:[channel,frameid]
def get_img_path(frame,img_path,channel):
    #get channel,frameid,carwheelbbox and carbbox value
    # data_channel = conform_data[1]
    # data_frameid = conform_data[0]
    # data_wb_index = conform_data[13]
    # data_cb_index = conform_data[14]
    if frame is None:
         return None
    channel = frame.iloc[0].loc['channel']
    frameid = frame.iloc[0].loc['frameid']

    imgs_name_list = []
    # img_path = front_path
    if channel == 0:
        front_path = img_path + "front" + "/"
        imgs_name_list = get_imgname_list(front_path)
        file_path = front_path
    elif channel == 1:
        left_path = img_path + "left" + "/"
        imgs_name_list = get_imgname_list(left_path)
        file_path = left_path
    elif channel == 3:
        right_path = img_path + "right" + "/"
        imgs_name_list = get_imgname_list(right_path)
        file_path = right_path

    matched_name = []
    
    for img_name in imgs_name_list:
        if int(float(frameid))==int(float(img_name[2])):
            matched_name = img_name
            break
        
    if matched_name:
        img_name_str = '_'.join(matched_name)
        img = img_name_str + '.bmp'
        img_path = file_path + img
        flag = True
        return img_path
    else:
        flag = False
        return None


            
# draw the frame bboxes which are matched on the ori img 
def draw_bbox(img,box,cls):
    """draw box on img by its cls: car:blue,front_wheel:yellow rear_wheel: pink_red"""
    left_top = (int(box[0]),int(box[1]))
    right_down = (int(box[2]),int(box[3]))

    if cls == 1:    #car
        color = (0,255,255) #BGR
    if cls == FRONT_WHEEL_CLS:    #front wheel
        color = (255,255,0)
    if cls == REAR_WHEEL_CLS:
        color = (255,0,255) 

    thickness = 1
    lineType = 8

    cv2.rectangle(img, left_top, right_down, color, thickness, lineType)


# frame_title=['frameid','channel','cls', \
#             'cb_x1','cb_y1','cb_x2''cb_y2', \
#             'wb_x1','wb_y1','wb_x2','wb_y2',\
#             'iou','cb_index','wb_index','carWheelCls','mode']
def draw_frame(img_path,frame):
    """draw all the boxes of frame on the img."""
    if img_path is None:
        return None
    img = cv2.imread(img_path)
    i = 0
    while i < len(frame):
        line = frame.iloc[i]
        if line.loc['mode'] == 1:
            car_bbox = (line.loc['cb_x1'],line.loc['cb_y1'],line.loc['cb_x2'],line.loc['cb_y2'])
            car_cls = 1
            carwheel_bbox = (line.loc['wb_x1'],line.loc['wb_y1'],line.loc['wb_x2'],line.loc['wb_y2'])
            carwheel_cls = line.loc['carWheelCls']
            draw_bbox(img,car_bbox,car_cls)
            draw_bbox(img,carwheel_bbox,carwheel_cls)
            i += 1
        else:
            i += 1
    return img
    

def save_frame(img,img_path,channel,frameid):
    """save the img to the path"""
    if channel == 0:
        file_path = img_path + "front" + "/"
        img_name = "tjp_" + "front_" + str(frameid) + "_" + "checked" + ".bmp"   #tjp_front_00001_checked.bmp
    elif channel == 1:
        file_path = img_path + "left" + "/"
        img_name = "tjp_" + "left_" + str(frameid) + "_" + "checked" + ".bmp"   #tjp_left_00001_checked.bmp
    elif channel == 3:
        file_path = img_path + "right" + "/"
        img_name = "tjp_" + "right_" + str(frameid) + "_" + "checked" + ".bmp"   #tjp_right_00001_checked.bmp

    img_path = file_path + img_name
    cv2.imwrite(img_path,img)








