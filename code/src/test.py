import pandas as pd 
import numpy as np

import relationship.car_carwheel as relation
import relationship.util as util

"""
test module
"""

def test():
    #import csv files
    #import front Wheel bbox csv 
    title=['frameid','channel','cls', \
                'cb_x1','cb_y1','cb_x2','cb_y2', \
                'wb_x1','wb_y1','wb_x2','wb_y2',\
                'iou','cb_index','wb_index','carWheelCls','mode']

    # ori_front_path = "../output/origin_img/front/"
    # ori_left_path = "../output/origin_img/left/"
    # ori_right_path = "../output/origin_img/right/"
    ori_img_path = "../output/origin_img/"

    # checked_front_path = "../output/checked_img/front/"
    # checked_left_path = "../output/checked_img/left/"
    # checked_right_path = "../output/checked_img/right/"
    checked_img_path = "../output/checked_img/"

    data_csv = pd.read_csv('../output/obd_rect_data.csv',header=None,sep=",")
    #add the title 
    data_csv.columns = title
    print("=================================[1][begin to check All the img ..]====================================\n")
    #generate frames 
    frames = util.gen_frames_from_csv_data(data_csv)

    checked_frame_data = pd.DataFrame()
    if frames is not None:
        for frame in frames:
            img_channel = frame.iloc[0].loc['channel']
            img_frameid = frame.iloc[0].loc['frameid']
            relationMap = relation.CarAndCarWheel(frame)
            relationMap.check(img_channel)
            print("-----------[frameid = {}, channel = {}]-----------\n".format(img_frameid,img_channel))
            print("relationMap = \n",relationMap.relationshipMatrix)
            new_frame = relationMap.update_frame()
            checked_frame_data = checked_frame_data.append(new_frame)
    else:
        print("generate frame from csv files : [the frame data of csv file is empty!!!]")
    
    checked_frame_data.to_csv('../output/checked_data_output.csv',encoding="utf-8-sig",mode='a',index=None)
    print("=================[car and carwheel matched work is over,and next begin to draw the check rect on the img..]===========")

    #conform checked_data manualy
    frames_checked = util.gen_frames_from_csv_data(checked_frame_data)
    
    #1.draw bbox on img
    if frames_checked is not None:
        for frame in frames_checked:
            img_channel = frame.iloc[0].loc['channel']
            img_frameid = frame.iloc[0].loc['frameid']
            img_path = util.get_img_path(frame,ori_img_path,img_channel)
            img = util.draw_frame(img_path,frame)
            if img is not None:
                util.save_frame(img,checked_img_path,img_channel,img_frameid)
    else:
        print("draw bbox on img : [the checked frame data is empty,can't find checked frame..]")
    print("=================[All the rect has been draw on the img,and next begin to secondary conform checked data frame by Compare the img..]===========")
    #2.compare the origin img and checked img
    matched_error_num = 0
    if frames_checked is not None:
        for frame in frames_checked:
            img_channel = frame.iloc[0].loc['channel']
            img_frameid = frame.iloc[0].loc['frameid']
            ori_path = util.get_img_path(frame,ori_img_path,img_channel)
            checked_path = util.get_img_path(frame,checked_img_path,img_channel)
            print("===>[ori_path = {}] and [checked_path = {}]".format(ori_path,checked_path))
            compareMap = util.CompareShow(ori_path,checked_path)
            compareMap.show()
            result = compareMap.get_matched_result()
            if not result:
                matched_error_num += 1
    else:
        print("compare the origin img and checked img : [the checked frame is empty...]")
    
    #3.calculate the error rate of the algrithm
    error_rate = matched_error_num/len(frames_checked)
    print("=======================[All frame checked and manually conformed over,the error_rate = {}]=======================".format(error_rate))





if __name__ == "__main__":
    test()

