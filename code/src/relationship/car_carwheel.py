# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

"""
module for detect the correct relationship between carWheel bbox and car bbox in a frame map. 
"""

import numpy as np 
import pandas as pd

import cv2



# car_wheel_BBox = (cls,id,pos)
#carBBox = (cls,id,pos)
#pos = (x1,y1,x2,y2)
#frame = ()

# frame_title=['frameid','channel','cls', \
#             'cb_x1','cb_y1','cb_x2''cb_y2', \
#             'wb_x1','wb_y1','wb_x2','wb_y2',\
#             'iou','cb_index','wb_index','carWheelCls','mode']
# carPos_title = ['x1','y1','x2','y2']
# carWheelPos_title = ['x1','y1','x2','y2']

# 
# frame = pd.DataFrame([],columns=frame_title)
# carPos = pd.Series([],index=carPos_title)
# carWheelPos = pd.Series([],index=carWheelPos)

#frame 
FRAME_WIDTH = 320
FRAME_HIGHT = 192

#object cls
FRONT_WHEEL_CLS = 7
REAR_WHEEL_CLS = 6
CAR_CLS = 1

#thresholds
BOTTOM_DIST_VALUE = 7



class CarAndCarWheel(object):
    "base bounding box object in one frame,frame is a pandas DataFrame"
    def __init__(self,frame): 
        self.frame = frame 
        self.frameid = frame.iloc[0].loc['frameid']
        self.channel = frame.iloc[0].loc['channel']
        self.car_BBoxes = self.gen_car_BBoxes() 
        self.car_wheel_BBoxes = self.gen_car_wheel_BBoxes()
        self.pred_modes = self.gen_pred_modes()
        self.relationshipMatrix = self.gen_relationship_matrix()
        
    def gen_car_BBoxes(self):
        car_BBoxes = []
        i = 0
        while i < len(self.frame):
            frame_data = self.frame.iloc[i]
            carPos = (frame_data.loc['cb_x1'], frame_data.loc['cb_y1'],frame_data.loc['cb_x2'],frame_data.loc['cb_y2'])    
            carWheelpos = (frame_data.loc['wb_x1'], frame_data.loc['wb_y1'],frame_data.loc['wb_x2'],frame_data.loc['wb_y2'])
            car_BBox = (frame_data.loc['cb_index'],frame_data.loc['cls'],carPos)
            car_BBoxes.append(car_BBox)
            i += 1
        return car_BBoxes

    def gen_car_wheel_BBoxes(self):
        car_wheel_BBoxes = []
        i = 0
        while i < len(self.frame):
            frame_data = self.frame.iloc[i]
            carPos = (frame_data.loc['cb_x1'], frame_data.loc['cb_y1'],frame_data.loc['cb_x2'],frame_data.loc['cb_y2'])    
            carWheelpos = (frame_data.loc['wb_x1'], frame_data.loc['wb_y1'],frame_data.loc['wb_x2'],frame_data.loc['wb_y2'])
            car_wheel_BBox = (frame_data.loc['wb_index'],frame_data.loc['carWheelCls'],carWheelpos)
            car_wheel_BBoxes.append(car_wheel_BBox)
            i += 1
        return car_wheel_BBoxes        

    def gen_pred_modes(self):
        pred_modes = []
        i = 0
        while i < len(self.frame):
            frame_data = self.frame.iloc[i]
            carPos = (frame_data.loc['cb_x1'], frame_data.loc['cb_y1'],frame_data.loc['cb_x2'],frame_data.loc['cb_y2'])    
            carWheelpos = (frame_data.loc['wb_x1'], frame_data.loc['wb_y1'],frame_data.loc['wb_x2'],frame_data.loc['wb_y2'])
            
            car_BBox = (frame_data.loc['cb_index'],frame_data.loc['cls'],carPos)
            car_wheel_BBox = (frame_data.loc['wb_index'],frame_data.loc['carWheelCls'],carWheelpos)

            predMode = (car_BBox[0],car_wheel_BBox[0],frame_data.loc['mode'])
            pred_modes.append(predMode)
            i += 1
        return pred_modes         

        

    def gen_relationship_matrix(self):
        # matrixRows = len(self.car_BBoxes)
        # matrixCols = len(self.car_wheel_BBoxes)
        # matrixRows = matrixCols = car_BBoxNum + car_wheel_BBoxNum

        title_idx = []
        title_col = []
        for car in self.car_BBoxes:
            if str(int(car[0])) not in title_idx:
                title_idx.append(str(int(car[0])))
        
        for carWheel in self.car_wheel_BBoxes:
            if str(int(carWheel[0])) not in title_col:
                title_col.append(str(int(carWheel[0])))

        matrixRows = len(title_idx)  
        matrixCols = len(title_col)  

        matrixValue = np.zeros((matrixRows,matrixCols))
        relationship_map = pd.DataFrame(matrixValue,columns=title_col,index=title_idx)
        

        for mode in self.pred_modes:
            if mode[2] == 1:
                relationship_map.loc[str(int(mode[0])),str(int(mode[1]))] = 1

        return relationship_map


   
    
    """
    base matheds for check
    """
    def calcu_area(self,bbox):
        return abs(bbox[2][2] - bbox[2][0]) * abs(bbox[2][3] - bbox[2][1])


    def calcu_iou(self,car_bbox,carwheel_bbox):
        "calculate the iou between car and carwheel"
        car_bbox_x1 = max(car_bbox[2][0],0)
        car_bbox_y1 = max(car_bbox[2][1],0)
        car_bbox_x2 = min(car_bbox[2][2],FRAME_WIDTH)
        car_bbox_y2 = min(car_bbox[2][3],FRAME_HIGHT)

        carwheel_bbox_x1 = max(carwheel_bbox[2][0],0)
        carwheel_bbox_y1 = max(carwheel_bbox[2][1],0)
        carwheel_bbox_x2 = min(carwheel_bbox[2][2],FRAME_WIDTH)
        carwheel_bbox_y2 = min(carwheel_bbox[2][3],FRAME_HIGHT)
 
        iou_x_min = max(car_bbox_x1,carwheel_bbox_x1)
        iou_x_max = max(car_bbox_x2,carwheel_bbox_x2)
        iou_y_min = min(car_bbox_y1,carwheel_bbox_y1)
        iou_y_max = min(car_bbox_y2,carwheel_bbox_y2)

        inter_size = (iou_x_max - iou_x_min + 1) * (iou_y_max - iou_y_min + 1)
        whole_size = (carwheel_bbox_x2 - carwheel_bbox_x1 + 1) * (carwheel_bbox_y2 - carwheel_bbox_y1 + 1)

        iou = inter_size/whole_size
        if car_bbox_x1 > carwheel_bbox_x2 or car_bbox_x2 < carwheel_bbox_x1 or car_bbox_y1 >carwheel_bbox_y2 or car_bbox_y2 < carwheel_bbox_y1:
            iou =  0.0

        return iou


    #modify carwheel cls value  front_cls = 7,rear_cls = 6
    def modify_carwheel_cls(self,car_bbox,carwheel_bbox,cls_value):
        i=0
        while i < len(self.frame):
            line = self.frame.iloc[i]
            if line.loc['cb_index'] == car_bbox[0] and line.loc['wb_index'] == carwheel_bbox[0]:
                line.loc['carWheelCls'] = cls_value
                break
            else:
                i += 1

    def get_carwheel_index_from_matrix(self,car,carwheel_list):
        carwheel_id = []
        for carwheel in carwheel_list:
            if self.relationshipMatrix.loc[str(int(car[0])),str(int(carwheel[0]))] == 1:
                carwheel_id.append(carwheel[0])
        return carwheel_id



#==============================================================================================================
    """
    Screening strategy function
    """
    #Screening strategy function
    #1.travese carWheel to conform one carWheel only can match one car_BBox.
    def carWheel_based_check(self,channel):
        if channel == 0:
            self.carWheel_based_check_front(channel)
        elif channel == 1:
            self.carWheel_based_check_left(channel)
        elif channel == 3:
            self.carWheel_based_check_right(channel)
        else:
            return None

    
    def carWheel_based_check_front(self,channel):

        for carWheel in self.car_wheel_BBoxes:
            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 0:
                continue
            
            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 1:
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        
                        if abs(car[2][0] - 0.5*(carWheel[2][2]-carWheel[2][0])) < 5:
                            # self.relationshipMatrix.loc[str(int(dist[0])),str(int(dist[1]))] = 0
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0


            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() > 1:
            # there are 2 car_BBox iou >0.5 with car_wheel_BBox ,so it should be remove one.
                bottomDist = []
                minDist = []
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                        dist = [car[0],carWheel[0],abs(car[2][3]- carWheel[2][3])]
                        bottomDist.append(dist)
                        minDist.append(dist[2])

                minIdx = minDist.index(min(minDist))
                i=0
                while i < len(bottomDist):
                    if i == minIdx:
                        if minDist[minIdx] < BOTTOM_DIST_VALUE:
                            i += 1
                        else:
                            self.relationshipMatrix.loc[str(int(bottomDist[i][0])),str(int(bottomDist[i][1]))] = 0
                            i += 1
                        # continue
                    else:
                        self.relationshipMatrix.loc[str(int(bottomDist[i][0])),str(int(bottomDist[i][1]))] = 0
                        print("========>",str(int(bottomDist[i][0])),str(int(bottomDist[i][1])))
                        i += 1


    def carWheel_based_check_left(self,channel):
    
        for carWheel in self.car_wheel_BBoxes:
            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 0:
                continue
            
            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 1:
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                        #iou check :iou >0.8
                        if self.calcu_iou(car,carWheel) < 0.8:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        
                        #dist check : dist >7
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

                        #width rate check: width_carwheel/width_car > 0.1
                        if abs(carWheel[2][2]- carWheel[2][0])/abs(car[2][2]- car[2][0]) < 0.1: 
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0


            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() > 1:
            # there are 2 car_BBox iou >0.5 with car_wheel_BBox ,so it should be remove one.
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                        #iou check :iou >0.8
                        if self.calcu_iou(car,carWheel) < 0.8:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        
                        #dist check : dist >7
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

                        #width rate check: width_carwheel/width_car > 0.1
                        if abs(carWheel[2][2]- carWheel[2][0])/abs(car[2][2]- car[2][0]) < 0.1: 
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() > 1:
                bottomDist = []
                minDist = []
                for car in self.car_BBoxes:
                        dist = [car[0],carWheel[0],abs(car[2][3]- carWheel[2][3])]
                        iou = [car[0],carWheel[0],self.calcu_iou(car,carWheel)]
                        bottomDist.append(dist)
                        minDist.append(dist[2])
    
                        minIdx = minDist.index(min(minDist))
                        i=0
                        while i < len(bottomDist):
                            if i == minIdx:
                                i += 1
                                # continue
                            else:
                                self.relationshipMatrix.loc[str(int(bottomDist[i][0])),str(int(bottomDist[i][1]))] = 0
                                i += 1
            elif self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 1:
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                    #iou check :iou >0.8
                        if self.calcu_iou(car,carWheel) < 0.8:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

                        #dist check : dist >7
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        #width rate check: width_carwheel/width_car > 0.1
                        if abs(carWheel[2][2]- carWheel[2][0])/abs(car[2][2]- car[2][0]) < 0.1: 
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0


    def carWheel_based_check_right(self,channel):
        
        for carWheel in self.car_wheel_BBoxes:
            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 0:
                continue
            
            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 1:
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                        #iou check :iou >0.8
                        if self.calcu_iou(car,carWheel) < 0.8:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        
                        #dist check : dist >7
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

                        #width rate check: width_carwheel/width_car > 0.1
                        if abs(carWheel[2][2]- carWheel[2][0])/abs(car[2][2]- car[2][0]) < 0.1: 
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0


            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() > 1:
            # there are 2 car_BBox iou >0.5 with car_wheel_BBox ,so it should be remove one.
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                        #iou check :iou >0.8
                        if self.calcu_iou(car,carWheel) < 0.8:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        
                        #dist check : dist >7
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

                        #width rate check: width_carwheel/width_car > 0.1
                        if abs(carWheel[2][2]- carWheel[2][0])/abs(car[2][2]- car[2][0]) < 0.1: 
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

            if self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() > 1:
                bottomDist = []
                minDist = []
                for car in self.car_BBoxes:
                        dist = [car[0],carWheel[0],abs(car[2][3]- carWheel[2][3])]
                        iou = [car[0],carWheel[0],self.calcu_iou(car,carWheel)]
                        bottomDist.append(dist)
                        minDist.append(dist[2])
    
                        minIdx = minDist.index(min(minDist))
                        i=0
                        while i < len(bottomDist):
                            if i == minIdx:
                                i += 1
                                # continue
                            else:
                                self.relationshipMatrix.loc[str(int(bottomDist[i][0])),str(int(bottomDist[i][1]))] = 0
                                i += 1
            elif self.relationshipMatrix.loc[:,str(int(carWheel[0]))].sum() == 1:
                for car in self.car_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] == 1.0:
                    #iou check :iou >0.8
                        if self.calcu_iou(car,carWheel) < 0.8:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

                        #dist check : dist >7
                        if abs(car[2][3]- carWheel[2][3]) > BOTTOM_DIST_VALUE:
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0
                        #width rate check: width_carwheel/width_car > 0.1
                        if abs(carWheel[2][2]- carWheel[2][0])/abs(car[2][2]- car[2][0]) < 0.1: 
                            self.relationshipMatrix.loc[str(int(car[0])),str(int(carWheel[0]))] = 0

        # print("carWheel based Screening has been travese!!!")

    def one_carWheel_check(self,car_bbox,carwheel_bbox,channel):
        "one carbbox only has one car_wheel"
        if abs(car_bbox[2][3] - carwheel_bbox[2][3]) > BOTTOM_DIST_VALUE:
            self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox[0]))] = 0

        if self.calcu_area(carwheel_bbox)/self.calcu_area(car_bbox) > 0.2 or self.calcu_area(carwheel_bbox)/self.calcu_area(car_bbox) < 0.01:
            self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox[0]))] = 0
        
        if carwheel_bbox[2][0] - car_bbox[2][0] < BOTTOM_DIST_VALUE and channel == 3:
                self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox[0]))] = 0

    def two_carWheel_check(self,car_bbox,carwheel_bbox1,carwheel_bbox2,channel):
        """one carbbox only has two car_wheel"""
        #0. all the carwheel_bbox should satify condition 1:one carbbox .
        self.one_carWheel_check(car_bbox,carwheel_bbox1,channel)
        self.one_carWheel_check(car_bbox,carwheel_bbox2,channel)
        #1.the cls same and cross with eachother.
        if carwheel_bbox1[1] == carwheel_bbox2[1] and self.calcu_iou(carwheel_bbox1,carwheel_bbox2) > 0:
            dist1 = abs(car_bbox[2][3] - carwheel_bbox1[2][3])
            dist2 = abs(car_bbox[2][3] - carwheel_bbox2[2][3])

            if dist1 < dist2:
                self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox2[0]))] = 0
            if dist2 < dist1:
                self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox1[0]))] = 0
        
        #2.the cls same and there is no cross with each other
        if carwheel_bbox1[1] == carwheel_bbox2[1] and self.calcu_iou(carwheel_bbox1,carwheel_bbox2) == 0:
            if self.calcu_area(carwheel_bbox1) > self.calcu_area(carwheel_bbox2):
                self.modify_carwheel_cls(car_bbox,carwheel_bbox1,FRONT_WHEEL_CLS)
                self.modify_carwheel_cls(car_bbox,carwheel_bbox2,REAR_WHEEL_CLS)
            if self.calcu_area(carwheel_bbox1) < self.calcu_area(carwheel_bbox2):
                self.modify_carwheel_cls(car_bbox,carwheel_bbox1,REAR_WHEEL_CLS)
                self.modify_carwheel_cls(car_bbox,carwheel_bbox2,FRONT_WHEEL_CLS)
                   
    def multi_carWheel_check(self,car_bbox,carwheel_list,channel):
        "one carbbox have more than two carwheel bboxes"
        if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() > 2.0:
            for car_wheel in carwheel_list:
                self.one_carWheel_check(car_bbox,car_wheel,channel)

        if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() > 2.0:
            carwheel_bbox_ids = self.get_carwheel_index_from_matrix(car_bbox,carwheel_list)
            front_idx = []
            rear_idx = []
            for carwheel in carwheel_list:
                if int(carwheel[1]) == 7:
                    front_idx.append(carwheel)
                if int(carwheel[1]) == 6:
                    rear_idx.append(carwheel)
            if len(front_idx) >= 2:
                mindist_front = []
                dist_front = []
                for frontwheel in front_idx:
                    if abs(frontwheel[2][3] - car_bbox[2][3]) > 7:
                        self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(frontwheel[0]))] = 0
                    else:
                        mindist_front.append(frontwheel)
                        dist_front.append(abs(frontwheel[2][3] - car_bbox[2][3]))

                if len(mindist_front)>=2:
                    mindist_idx = dist_front.index(min(dist_front))
                    i=0
                    while i<len(mindist_front):
                        if i == mindist_idx:
                            i += 1
                            continue
                        else:
                            self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(mindist_front[i][0]))] = 0
                            i += 1

            if len(rear_idx) >= 2:
                mindist_rear = []
                dist_rear = []
                for rearwheel in rear_idx:
                    if abs(rearwheel[2][3] - car_bbox[2][3]) > 7:
                        self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(rearwheel[0]))] = 0
                    else:
                        mindist_rear.append(rearwheel)
                        dist_rear.append(abs(rearwheel[2][3] - car_bbox[2][3]))

                if len(mindist_rear)>=2:
                    mindist_idx = dist_rear.index(min(dist_rear))
                    i=0
                    while i<len(mindist_rear):
                        if i == mindist_idx:
                            i += 1
                            continue
                        else:
                            self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(mindist_rear[i][0]))] = 0     
                            i += 1  
        
        if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() ==2.0:
            carwheel_bbox_ids = self.get_carwheel_index_from_matrix(car_bbox,carwheel_list)
            for carwheel in carwheel_list:
                if carwheel[0] == carwheel_bbox_ids[0]:
                    carwheel_bbox1 = carwheel
                if carwheel[0] == carwheel_bbox_ids[1]:
                    carwheel_bbox2 = carwheel

            self.two_carWheel_check(car_bbox,carwheel_bbox1,carwheel_bbox2,int(channel))

#==============================================================================================================
    def check(self,channel):
        #1. carWheel based check.
        self.carWheel_based_check(int(channel))

        title_idx = []
        for car in self.car_BBoxes:
            title_idx.append(str(car[0]))
        
        #2. car based check
        carwheel_bbox_two = []
        carwheel_bbox_multi = []
        for car_bbox in self.car_BBoxes:
            if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() == 0:
                continue
            if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() == 1.0:
                for carwheel_bbox in self.car_wheel_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox[0]))] == 1.0:
                        self.one_carWheel_check(car_bbox,carwheel_bbox,int(channel))
            
            if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() == 2.0:
                for carwheel_bbox in self.car_wheel_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox[0]))] == 1.0:
                        carwheel_bbox_two.append(carwheel_bbox)

                carwheel_bbox1 = carwheel_bbox_two[0]
                carwheel_bbox2 = carwheel_bbox_two[1]
                self.two_carWheel_check(car_bbox,carwheel_bbox1,carwheel_bbox2,int(channel))

            if self.relationshipMatrix.loc[str(int(car_bbox[0]))].sum() > 2.0:
                for carwheel_bbox in self.car_wheel_BBoxes:
                    if self.relationshipMatrix.loc[str(int(car_bbox[0])),str(int(carwheel_bbox[0]))] == 1.0:
                        carwheel_bbox_multi.append(carwheel_bbox)

                self.multi_carWheel_check(car_bbox,carwheel_bbox_multi,int(channel))
        
        #3. carWheel based check.
        # self.carWheel_based_check()


    def update_frame(self):
        assert(len(self.car_BBoxes)>0 and len(self.car_wheel_BBoxes)>0)
        for car_bbox in self.car_BBoxes:
            for carwheel_bbox in self.car_wheel_BBoxes:
                car_id,carwheel_id = car_bbox[0],carwheel_bbox[0]
                i = 0
                while i < len(self.frame):
                    line = self.frame.iloc[i]
                    if (line.loc['cb_index'],line.loc['wb_index']) == (car_id,carwheel_id):
                        line.loc['mode'] = self.relationshipMatrix.loc[str(int(car_id)),str(int(carwheel_id))]
                    i += 1
    
        return self.frame
#=======================================================================================================================================





 

        




