# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import copy
import heapq

# https://www.cnblogs.com/jingsupo/p/python_curve_fit.html
# line_1_flag = 1
parking_slot_lane_flag = 1
line_2_flag = 2
center_lane_flag = 8
mode = 'ASP'



def mkdir(path):
    if (os.path.exists(path) == False):
        os.mkdir(path)
    else:
        pass

def line_fit_show_y(x, y, n=1):  # 3th Polynom 
    # z1 = np.polyfit(x, y, n) # 用3次多项式拟合
    # z1 是一个向量，包含四个值，表示(ax^3+bx^2+cx+d)
    z1 = np.polyfit(y, x, n)  # 用3次多项式拟合
    # 将系数带入方程，得到一个函数式子 p1
    p1 = np.poly1d(z1)
    # xvals = get_fit_x_according_y(y,p1,x)
    # print(p1) # 在屏幕上打印拟合多项式
    xvals = p1(y)  # 也可以使用yvals=np.polyval(z1,x)
    # print(p1)
    return p1, y, xvals

def color_table():
    color_list = [[0, 255, 0], [0, 10, 255], [255, 0, 0]]
    return color_list

def draw_line_in_image(image, x_value, y_value, point_color):
    high, width, c = image.shape
    draw_step = 3
    # point_color = color_table()[1]
    line_color = color_table()[2]
    # --------------------fit with point----------------------------
    point_size = 2
    thickness = 2

    for i in range(len(x_value)):
        if (y_value[i] >= high or y_value[i] <= 0):
            continue
        if (i % draw_step != 0):
            continue
        point = (int(x_value[i]), int(y_value[i]))
        cv2.circle(image, point, point_size, point_color, thickness)
    return image



def calcu_area_line(class11_line):
    d_and_index = []
    class11_line = np.array(class11_line)
    # print(class11_line)
    for index1,X1 in enumerate(class11_line[:,:2]):
        for index2,X2 in enumerate(class11_line[:,2:]):
            d = ((X1[0]-X2[0])**2 + (X1[1]-X2[1])**2)**0.5
            d_and_index.append([d,index1,index2])
    d_and_index = np.array(d_and_index)
    # print("d_and_index",d_and_index)
    where_max_d = np.where(d_and_index[:,0] == max(d_and_index[:,0]))[0]
    index = d_and_index[where_max_d,1:][0]
    line = [class11_line[int(index[0])][:2], class11_line[int(index[1])][2:]]
    line = np.array(line)
    return line



def get_fit_point(edge, p1, p2):
    high, width = edge.shape
    value_flag = 255
    # print(p1,p2)
    if (p1[0]-p2[0]) != 0:
        k = (p1[1]-p2[1])/(p1[0]-p2[0])
    else:
        k = float("inf")
    # print("k",k)
    abs_k = abs(k)
    if abs_k >= 1.5:    #windows along the y axis
        dy = 2
        w_x = 10
        if p1[1] < p2[1]:
            p_begin = p1
            p_end = p2
        else:
            p_begin = p2
            p_end = p1
        # number_windows = int(abs((p1[1]-p2[1]))/dy)
        dx = dy/k
        
        line_p_y = np.arange(int(p_begin[1]),int(p_end[1]),dy)
        line_p_x = []
        
        print("number of points",len(line_p_y))
        for i in range(len(line_p_y)):
            
            line_x = p_begin[0] + i * dx
            line_p_x.append(line_x)
        x_line_begin = [i - w_x    for i in line_p_x]
        x_line_end = [i + w_x   for i in line_p_x]
        begin_point_x = []
        begin_point_y = []
        for index,row in enumerate(line_p_y):
            for col in range(int(x_line_begin[index]), int(x_line_end[index]),1):
                if col < 480:
                    if (edge[row,col] == value_flag):
                        # point = [row,col]
                        # begin_point.append(point)
                        begin_point_x.append(col)
                        begin_point_y.append(row)
                        break
        end_point_x = []
        end_point_y = []
        for index,row in enumerate(line_p_y):
            for col in range(int(x_line_end[index]), int(x_line_begin[index]),-1):
                if col < 480:
                    if (edge[row,col] == value_flag):
                    # print(col)
                                                #从上到下8个像素踩一个点，从右边步长为1开始历遍，当第一个值为255也就是车道线的语义分割时，BREAK结束，故采样的是右边的点，
                        # point = [row,col]
                        # end_point.append(point)
                        end_point_x.append(col)
                        end_point_y.append(row)
                        break
        # print("extract points ***************************************")
        # print([begin_point_x, begin_point_y, end_point_x, end_point_y])
        return [begin_point_x, begin_point_y, end_point_x, end_point_y]

    else:    #windows along the x axis
        dx = 2
        w_y = 10
        if p1[0] < p2[0]:
            p_begin = p1
            p_end = p2
        else:
            p_begin = p2
            p_end = p1
        # number_windows = int(abs((p1[0]-p2[0]))/dx)
        dy = k * dx
        
        line_p_x = np.arange(int(p_begin[0]),int(p_end[0]),dx)
        line_p_y = []
        print("number of points",len(line_p_x))
        for i in range(len(line_p_x)):
            line_y = p_begin[1] + i * dy
            line_p_y.append(line_y)
        y_line_begin = [i - w_y    for i in line_p_y]
        y_line_end = [i + w_y   for i in line_p_y]
        begin_point_x = []
        begin_point_y = []
        for index,col in enumerate(line_p_x):
            for row in range(int(y_line_begin[index]), int(y_line_end[index]),1):
                if row < 600:
                    if (edge[row,col] == value_flag):
                        # point = [row,col]
                        # begin_point.append(point)
                        begin_point_x.append(col)
                        begin_point_y.append(row)
                        break
        end_point_x = []
        end_point_y = []
        for index,col in enumerate(line_p_x):
            for row in range(int(y_line_end[index]), int(y_line_begin[index]),-1):
                if row < 600:
                    if (edge[row,col] == value_flag):
                    # print(col)
                                                #从上到下8个像素踩一个点，从右边步长为1开始历遍，当第一个值为255也就是车道线的语义分割时，BREAK结束，故采样的是右边的点，
                        # point = [row,col]
                        # end_point.append(point)
                        end_point_x.append(col)
                        end_point_y.append(row)
                        break
        # print("extract points ***************************************")
        # print([begin_point_x, begin_point_y, end_point_x, end_point_y])
        return [begin_point_x, begin_point_y, end_point_x, end_point_y]

def get_fit_point_1_or_2(edge,class11_12):
    if len(class11_12) == 1:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_12 = [line_point_11_1]
        
    if len(class11_12) == 2:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_2 = get_fit_point(edge,class11_12[1][0],class11_12[1][1])
        line_point_11_12 = [line_point_11_1,line_point_11_2]
    if len(class11_12) == 3:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_2 = get_fit_point(edge,class11_12[1][0],class11_12[1][1])
        line_point_11_3 = get_fit_point(edge,class11_12[2][0],class11_12[2][1])
        line_point_11_12 = [line_point_11_1,line_point_11_2,line_point_11_3]
    if len(class11_12) == 4:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_2 = get_fit_point(edge,class11_12[1][0],class11_12[1][1])
        line_point_11_3 = get_fit_point(edge,class11_12[2][0],class11_12[2][1])
        line_point_11_4 = get_fit_point(edge,class11_12[3][0],class11_12[3][1])
        line_point_11_12 = [line_point_11_1,line_point_11_2,line_point_11_3,line_point_11_4]
    if len(class11_12) == 5:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_2 = get_fit_point(edge,class11_12[1][0],class11_12[1][1])
        line_point_11_3 = get_fit_point(edge,class11_12[2][0],class11_12[2][1])
        line_point_11_4 = get_fit_point(edge,class11_12[3][0],class11_12[3][1])
        line_point_11_5 = get_fit_point(edge,class11_12[4][0],class11_12[4][1])
        line_point_11_12 = [line_point_11_1,line_point_11_2,line_point_11_3,line_point_11_4, line_point_11_5]
    if len(class11_12) == 6:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_2 = get_fit_point(edge,class11_12[1][0],class11_12[1][1])
        line_point_11_3 = get_fit_point(edge,class11_12[2][0],class11_12[2][1])
        line_point_11_4 = get_fit_point(edge,class11_12[3][0],class11_12[3][1])
        line_point_11_5 = get_fit_point(edge,class11_12[4][0],class11_12[4][1])
        line_point_11_6 = get_fit_point(edge,class11_12[5][0],class11_12[5][1])
        line_point_11_12 = [line_point_11_1,line_point_11_2,line_point_11_3,line_point_11_4, line_point_11_5, line_point_11_6]
    if len(class11_12) == 7:
        line_point_11_1 = get_fit_point(edge,class11_12[0][0],class11_12[0][1])
        line_point_11_2 = get_fit_point(edge,class11_12[1][0],class11_12[1][1])
        line_point_11_3 = get_fit_point(edge,class11_12[2][0],class11_12[2][1])
        line_point_11_4 = get_fit_point(edge,class11_12[3][0],class11_12[3][1])
        line_point_11_5 = get_fit_point(edge,class11_12[4][0],class11_12[4][1])
        line_point_11_6 = get_fit_point(edge,class11_12[5][0],class11_12[5][1])
        line_point_11_7 = get_fit_point(edge,class11_12[6][0],class11_12[6][1])
        line_point_11_12 = [line_point_11_1,line_point_11_2,line_point_11_3,line_point_11_4, line_point_11_5, line_point_11_6, line_point_11_7]

    return line_point_11_12


def divide_by_distance(class_line,class_vector):
    class_p = class_line[0][:2]
    class_point_to_line = []
    D = []
    for index,line in enumerate(class_line):
        a = np.array([line[0]-class_p[0],line[1]-class_p[1]])
        b = class_vector[index]
        c = (a.dot(b)) * b
        e = a - c
        d = (e[0]**2 + e[1]**2)**0.5
        D.append(d)
       
    # print("D",D)
    D = np.array(D)
    class1_index = np.where(D<16)[0]
    class1_line = class_line[class1_index]
    # print("class1_index",class1_index)
    # print("class1_line",class1_line)
    if len(D)==len(class1_index):
        return [class1_line]
    if len(D)>len(class1_index):
        class2_index = np.where(D>=16)[0]
        class2_line = class_line[class2_index]
        # print("class2_index",class2_index)
        # print("class2_line",class2_line)
        return [class1_line,class2_line]

def unit_vector(line):
    Line = list(line)
    vecs = []
    for x1,y1,x2,y2 in Line:
        d = ((x2-x1)**2 + (y1-y2)**2)**(1/2)
        vecs.append([(x2-x1)/d, (y2-y1)/d])
    vecs = np.array(vecs)
    return vecs


def divide_paralle_line(class11_line,img,edge,color1,color2,image_ori):
    # print(len(class11_line))
    vecs = unit_vector(class11_line)
    class11_line = np.array(class11_line)
    # print("sa",vecs)
    class1 = divide_by_distance(class11_line,vecs)

    if len(class1) == 1:
        Line_all = class1

    print("sad",len(class1))
    if len(class1) == 2:
        c1 = class1[0]
        c2 = class1[1]
        v1 = unit_vector(c1)
        l1 = divide_by_distance(c1,v1)
        print("l1", len(l1))
        v2 = unit_vector(c2)
        l2 = divide_by_distance(c2,v2)
        print("l2", len(l2))

        if len(l1) == 1 and len(l2) == 1:
            Line_all = [l1[0], l2[0]]
        elif len(l1) == 2 and len(l2) == 1:
            Line_all = [l1[0], l1[1], l2[0]]
        elif len(l1) == 1 and len(l2) == 2:
            Line_all = [l1[0], l2[0], l2[1]]
        elif len(l1) == 2 and len(l2) == 2:
            Line_all = [l1[0], l1[1], l2[0], l2[1]]



    line = calcu_area_line(class11_line)
    print("line",line)

    l1 = class11_line[0]
    lx1 = l1[0]
    ly1 = l1[1]
    lx2 = l1[2]
    ly2 = l1[3]

    if (lx1-lx2) != 0:
        lk = (ly1-ly2)/(lx1-lx2)
    else:
        lk = float("inf")
    print("lk",lk)

    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]
    # print("line",line)
    if (x1-x2) != 0:
        k = (y1-y2)/(x1-x2)
    else:
        k = float("inf")
    print("k",k)

    
    # cv2.line(img,(x1,y1),(x2,y2),(100,100,),3)
    cx,cy = [int((x1+x2)/2), int((y1+y2)/2)]
    # print("cx",cx,"cy",cy)
    cv2.circle(img,(cx,cy),6,(10,0,200),15)

    a = np.array([lx1 - cx, ly1 - cy])
    b1 = np.array([lx1-lx2, ly1 - ly2])
    sca_b = (b1[0]**2 + b1[1]**2)**(1/2)
    b = np.array([b1[0]/sca_b, b1[1]/sca_b])
    c = (a.dot(b)) * b
    e = a - c
    d_ab = (e[0]**2 + e[1]**2)**0.5
    print("cx:",cx,"cy:",cy)
    print("d seperate:", d_ab)
    roi = edge[cy-8:cy+8,cx-8:cx+8]/255
    sum_roi = np.sum(np.sum(roi))
    print("sum_roi",sum_roi)
    print("**************************")

    # if (sum_roi == 0) or (d_ab >= 25) : #or ((abs(k - lk)>1.5) and (lk != float("inf"))) :
    #     if d_ab < 25:
    #         if abs(lk) >= 2: # (abs(lk) > 2) or ((abs(lk) <= 1) and (d_ab >= 25)) :
    #             class111_line = []
    #             class112_line = []
    #             for x1,y1,x2,y2 in class11_line:
    #                 if cy >= (y1 + y2)/2:
    #                     class111_line.append([x1,y1,x2,y2])
    #                 if cy < (y1 + y2)/2:
    #                     class112_line.append([x1,y1,x2,y2])
    #         else:            
    #             class111_line = []
    #             class112_line = []
    #             for x1,y1,x2,y2 in class11_line:
    #                 if cx >= (x1 + x2)/2:
    #                     class111_line.append([x1,y1,x2,y2])
    #                 if cx < (x1 + x2)/2:
    #                     class112_line.append([x1,y1,x2,y2])
    #     else:  #(abs(lk) <= 2) or ((abs(lk) > 1) and (d_ab >= 25)):
    #         if abs(lk) <= 1: # (abs(lk) > 2) or ((abs(lk) <= 1) and (d_ab >= 25)) :
    #             class111_line = []
    #             class112_line = []
    #             for x1,y1,x2,y2 in class11_line:
    #                 if cy >= (y1 + y2)/2:
    #                     class111_line.append([x1,y1,x2,y2])
    #                 if cy < (y1 + y2)/2:
    #                     class112_line.append([x1,y1,x2,y2])
    #         else:            
    #             class111_line = []
    #             class112_line = []
    #             for x1,y1,x2,y2 in class11_line:
    #                 if cx >= (x1 + x2)/2:
    #                     class111_line.append([x1,y1,x2,y2])
    #                 if cx < (x1 + x2)/2:
    #                     class112_line.append([x1,y1,x2,y2])  
    #     class12 = [class111_line,class112_line]
    #     print(class12)
    #     if [] not in class12:
    #         class11_12 = [class111_line,class112_line]
    #     if class12[0] == []:
    #         class11_12 = [class12[1]]
    #     if class12[1] == []:
    #         class11_12 = [class12[0]]


    # else:
        
    #     class11_12 = [class11_line]

    
    return Line_all



def divide_coaxis_line(class11_line,img,edge,color1,color2,color3,color4,image_ori):

    line = calcu_area_line(class11_line)

    l1 = class11_line[0]
    lx1 = l1[0]
    ly1 = l1[1]
    lx2 = l1[2]
    ly2 = l1[3]

    if (lx1-lx2) != 0:
        lk = (ly1-ly2)/(lx1-lx2)
    else:
        lk = float("inf")
    # print("lk",lk)

    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]
    # print("line",line)
    if (x1-x2) != 0:
        k = (y1-y2)/(x1-x2)
    else:
        k = float("inf")
    # print("k",k)
    
    # cv2.line(img,(x1,y1),(x2,y2),(100,100,),3)
    cx,cy = [int((x1+x2)/2), int((y1+y2)/2)]
    # print("cx",cx,"cy",cy)
    cv2.circle(img,(cx,cy),6,(10,0,200),15)

    a = np.array([lx1 - cx, ly1 - cy])
    b1 = np.array([lx1-lx2, ly1 - ly2])
    sca_b = (b1[0]**2 + b1[1]**2)**(1/2)
    b = np.array([b1[0]/sca_b, b1[1]/sca_b])
    c = (a.dot(b)) * b
    e = a - c
    d_ab = (e[0]**2 + e[1]**2)**0.5
    # print("cx:",cx,"cy:",cy)
    # print("d seperate:", d_ab)
    roi = edge[cy-9:cy+9,cx-9:cx+9]/255
    sum_roi = np.sum(np.sum(roi))
    # print("sum_roi",sum_roi)
    # print((sum_roi == 0) or (abs(k - lk)>2))
    if (sum_roi == 0 and len(class11_line)>1) or (d_ab >= 25 and len(class11_line)>1) : #or ((abs(k - lk)>1.5) and (lk != float("inf"))) :
        if d_ab < 25:
            if abs(lk) >= 2: # (abs(lk) > 2) or ((abs(lk) <= 1) and (d_ab >= 25)) :
                class111_line = []
                class112_line = []
                for x1,y1,x2,y2 in class11_line:
                    if cy >= (y1 + y2)/2:
                        class111_line.append([x1,y1,x2,y2])
                    if cy < (y1 + y2)/2:
                        class112_line.append([x1,y1,x2,y2])
            else:            
                class111_line = []
                class112_line = []
                for x1,y1,x2,y2 in class11_line:
                    if cx >= (x1 + x2)/2:
                        class111_line.append([x1,y1,x2,y2])
                    if cx < (x1 + x2)/2:
                        class112_line.append([x1,y1,x2,y2])
        else:  #(abs(lk) <= 2) or ((abs(lk) > 1) and (d_ab >= 25)):
            if abs(lk) <= 1: # (abs(lk) > 2) or ((abs(lk) <= 1) and (d_ab >= 25)) :
                class111_line = []
                class112_line = []
                for x1,y1,x2,y2 in class11_line:
                    if cy >= (y1 + y2)/2:
                        class111_line.append([x1,y1,x2,y2])
                    if cy < (y1 + y2)/2:
                        class112_line.append([x1,y1,x2,y2])
            else:            
                class111_line = []
                class112_line = []
                for x1,y1,x2,y2 in class11_line:
                    if cx >= (x1 + x2)/2:
                        class111_line.append([x1,y1,x2,y2])
                    if cx < (x1 + x2)/2:
                        class112_line.append([x1,y1,x2,y2])  
        if class111_line == []:
            vecs112 = unit_vector(class112_line)
            class112_line = np.array(class112_line)
            class11_12 = divide_by_distance(class112_line,vecs112)
        if class112_line == []:
            vecs111 = unit_vector(class111_line)
            class111_line = np.array(class111_line)
            class11_12 = divide_by_distance(class111_line,vecs111)           
        if class111_line != [] and class112_line != []:
            class11_12 = [class111_line,class112_line]

        # if [] not in class1_12:
        #     class11_12 = [class111_line,class112_line]
        # if class1_12[0] == []:
        #     class11_12 = [class1_12[1]]
        # if class1_12[1] == []:
        #     class11_12 = [class1_12[0]]
        print("class11_12",class11_12)
        class11_12_1 = divide_paralle_line(class11_12[0],img,edge,color1,color2,image_ori)
        
        class11_12_2 = divide_paralle_line(class11_12[1],img,edge,color3,color4,image_ori)

        if len(class11_12_1) == 1:
            if len(class11_12_2) == 1:
                class11_12_12 = [class11_12_1[0], class11_12_2[0]]
            if len(class11_12_2) == 2:
                class11_12_12 = [class11_12_1[0], class11_12_2[0], class11_12_2[1]]
            if len(class11_12_2) == 3:
                class11_12_12 = [class11_12_1[0], class11_12_2[0], class11_12_2[1], class11_12_2[2]]
            if len(class11_12_2) == 4:
                class11_12_12 = [class11_12_1[0], class11_12_2[0], class11_12_2[1], class11_12_2[2], class11_12_2[3]]    
        if len(class11_12_1) == 2:
            if len(class11_12_2) == 1:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_2[0]]
            if len(class11_12_2) == 2:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_2[0], class11_12_2[1]]
            if len(class11_12_2) == 3:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_2[0], class11_12_2[1], class11_12_2[2]]
            if len(class11_12_2) == 4:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_2[0], class11_12_2[1], class11_12_2[2], class11_12_2[3]]   
        if len(class11_12_1) == 3:
            if len(class11_12_2) == 1:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_2[0]]
            if len(class11_12_2) == 2:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_2[0], class11_12_2[1]]
            if len(class11_12_2) == 3:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_2[0], class11_12_2[1], class11_12_2[2]]
            if len(class11_12_2) == 4:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_2[0], class11_12_2[1], class11_12_2[2], class11_12_2[3]]   
        if len(class11_12_1) == 4:
            if len(class11_12_2) == 1:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_1[3], class11_12_2[0]]
            if len(class11_12_2) == 2:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_1[3], class11_12_2[0], class11_12_2[1]]
            if len(class11_12_2) == 3:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_1[3], class11_12_2[0], class11_12_2[1], class11_12_2[2]]
            if len(class11_12_2) == 4:
                class11_12_12 = [class11_12_1[0], class11_12_1[1], class11_12_1[2], class11_12_1[3], class11_12_2[0], class11_12_2[1], class11_12_2[2], class11_12_2[3]]   
            
        print("class11_12_12",len(class11_12_12))
        # class11_12 = [x for x in class11_12 if x]
        # print("asd",class11_12)
        
        if [] not in class11_12_12:
            if len(class11_12_12) == 2:
                class11_1_line = class11_12_12[0]
                line1 = calcu_area_line(class11_1_line)
                # print("line1",line1)
                x1 = line1[0][0]
                y1 = line1[0][1]
                x2 = line1[1][0]
                y2 = line1[1][1]
                # cx1 = int((x1 + x2)/2)
                # cy1 = int((y1 + y2)/2)
                # lane_color1 = image_ori[cy1,cx1]
                # print("lane_color1",lane_color1)
                cv2.line(img,(x1,y1),(x2,y2),color1,3)
                class11_2_line = class11_12_12[1]
                line2 = calcu_area_line(class11_2_line)
                x11 = line2[0][0]
                y11 = line2[0][1]
                x22 = line2[1][0]
                y22 = line2[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x11,y11),(x22,y22),color2,3)
                line12 = [line1, line2]
                return line12
            if len(class11_12_12) == 3:
                class11_1_line = class11_12_12[0]
                line1 = calcu_area_line(class11_1_line)
                # print("line1",line1)
                x1 = line1[0][0]
                y1 = line1[0][1]
                x2 = line1[1][0]
                y2 = line1[1][1]
                # cx1 = int((x1 + x2)/2)
                # cy1 = int((y1 + y2)/2)
                # lane_color1 = image_ori[cy1,cx1]
                # print("lane_color1",lane_color1)
                cv2.line(img,(x1,y1),(x2,y2),color1,3)
                class11_2_line = class11_12_12[1]
                line2 = calcu_area_line(class11_2_line)
                x11 = line2[0][0]
                y11 = line2[0][1]
                x22 = line2[1][0]
                y22 = line2[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x11,y11),(x22,y22),color2,3)
                class11_3_line = class11_12_12[2]
                line3 = calcu_area_line(class11_3_line)
                x111 = line3[0][0]
                y111 = line3[0][1]
                x222 = line3[1][0]
                y222 = line3[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x111,y111),(x222,y222),color3,3)
                line12 = [line1, line2, line3]
                return line12
            if len(class11_12_12) == 4:
                class11_1_line = class11_12_12[0]
                line1 = calcu_area_line(class11_1_line)
                # print("line1",line1)
                x1 = line1[0][0]
                y1 = line1[0][1]
                x2 = line1[1][0]
                y2 = line1[1][1]
                # cx1 = int((x1 + x2)/2)
                # cy1 = int((y1 + y2)/2)
                # lane_color1 = image_ori[cy1,cx1]
                # print("lane_color1",lane_color1)
                cv2.line(img,(x1,y1),(x2,y2),color1,3)
                class11_2_line = class11_12_12[1]
                line2 = calcu_area_line(class11_2_line)
                x11 = line2[0][0]
                y11 = line2[0][1]
                x22 = line2[1][0]
                y22 = line2[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x11,y11),(x22,y22),color2,3)
                class11_3_line = class11_12_12[2]
                line3 = calcu_area_line(class11_3_line)
                x111 = line3[0][0]
                y111 = line3[0][1]
                x222 = line3[1][0]
                y222 = line3[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x111,y111),(x222,y222),color3,3)
                class11_4_line = class11_12_12[3]
                line4 = calcu_area_line(class11_4_line)
                x1111 = line4[0][0]
                y1111 = line4[0][1]
                x2222 = line4[1][0]
                y2222 = line4[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x1111,y1111),(x2222,y2222),color4,3)
                line12 = [line1, line2, line3,line4]
                
                return line12

            if len(class11_12_12) == 5:
                class11_1_line = class11_12_12[0]
                line1 = calcu_area_line(class11_1_line)
                # print("line1",line1)
                x1 = line1[0][0]
                y1 = line1[0][1]
                x2 = line1[1][0]
                y2 = line1[1][1]
                # cx1 = int((x1 + x2)/2)
                # cy1 = int((y1 + y2)/2)
                # lane_color1 = image_ori[cy1,cx1]
                # print("lane_color1",lane_color1)
                cv2.line(img,(x1,y1),(x2,y2),color1,3)
                class11_2_line = class11_12_12[1]
                line2 = calcu_area_line(class11_2_line)
                x11 = line2[0][0]
                y11 = line2[0][1]
                x22 = line2[1][0]
                y22 = line2[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x11,y11),(x22,y22),color2,3)
                class11_3_line = class11_12_12[2]
                line3 = calcu_area_line(class11_3_line)
                x111 = line3[0][0]
                y111 = line3[0][1]
                x222 = line3[1][0]
                y222 = line3[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x111,y111),(x222,y222),color3,3)
                class11_4_line = class11_12_12[3]
                line4 = calcu_area_line(class11_4_line)
                x1111 = line4[0][0]
                y1111 = line4[0][1]
                x2222 = line4[1][0]
                y2222 = line4[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x1111,y1111),(x2222,y2222),color4,3)

                class11_5_line = class11_12_12[4]
                line5 = calcu_area_line(class11_5_line)
                x11111 = line5[0][0]
                y11111 = line5[0][1]
                x22222 = line5[1][0]
                y22222 = line5[1][1]
                cv2.line(img,(x11111,y11111),(x22222,y22222),color4,3)
                line12 = [line1, line2, line3, line4, line5]
                
                return line12

            if len(class11_12_12) == 6:
                class11_1_line = class11_12_12[0]
                line1 = calcu_area_line(class11_1_line)
                # print("line1",line1)
                x1 = line1[0][0]
                y1 = line1[0][1]
                x2 = line1[1][0]
                y2 = line1[1][1]
                # cx1 = int((x1 + x2)/2)
                # cy1 = int((y1 + y2)/2)
                # lane_color1 = image_ori[cy1,cx1]
                # print("lane_color1",lane_color1)
                cv2.line(img,(x1,y1),(x2,y2),color1,3)
                class11_2_line = class11_12_12[1]
                line2 = calcu_area_line(class11_2_line)
                x11 = line2[0][0]
                y11 = line2[0][1]
                x22 = line2[1][0]
                y22 = line2[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x11,y11),(x22,y22),color2,3)
                class11_3_line = class11_12_12[2]
                line3 = calcu_area_line(class11_3_line)
                x111 = line3[0][0]
                y111 = line3[0][1]
                x222 = line3[1][0]
                y222 = line3[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x111,y111),(x222,y222),color3,3)
                class11_4_line = class11_12_12[3]
                line4 = calcu_area_line(class11_4_line)
                x1111 = line4[0][0]
                y1111 = line4[0][1]
                x2222 = line4[1][0]
                y2222 = line4[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x1111,y1111),(x2222,y2222),color4,3)

                class11_5_line = class11_12_12[4]
                line5 = calcu_area_line(class11_5_line)
                x11111 = line5[0][0]
                y11111 = line5[0][1]
                x22222 = line5[1][0]
                y22222 = line5[1][1]
                cv2.line(img,(x11111,y11111),(x22222,y22222),color4,3)

                class11_6_line = class11_12_12[5]
                line6 = calcu_area_line(class11_6_line)
                x111111 = line6[0][0]
                y111111 = line6[0][1]
                x222222 = line6[1][0]
                y222222 = line6[1][1]
                cv2.line(img,(x111111,y111111),(x222222,y222222),color4,3)
                line12 = [line1, line2, line3,line4, line5, line6]
                
                return line12

            if len(class11_12_12) == 7:
                class11_1_line = class11_12_12[0]
                line1 = calcu_area_line(class11_1_line)
                # print("line1",line1)
                x1 = line1[0][0]
                y1 = line1[0][1]
                x2 = line1[1][0]
                y2 = line1[1][1]
                # cx1 = int((x1 + x2)/2)
                # cy1 = int((y1 + y2)/2)
                # lane_color1 = image_ori[cy1,cx1]
                # print("lane_color1",lane_color1)
                cv2.line(img,(x1,y1),(x2,y2),color1,3)
                class11_2_line = class11_12_12[1]
                line2 = calcu_area_line(class11_2_line)
                x11 = line2[0][0]
                y11 = line2[0][1]
                x22 = line2[1][0]
                y22 = line2[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x11,y11),(x22,y22),color2,3)
                class11_3_line = class11_12_12[2]
                line3 = calcu_area_line(class11_3_line)
                x111 = line3[0][0]
                y111 = line3[0][1]
                x222 = line3[1][0]
                y222 = line3[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x111,y111),(x222,y222),color3,3)
                class11_4_line = class11_12_12[3]
                line4 = calcu_area_line(class11_4_line)
                x1111 = line4[0][0]
                y1111 = line4[0][1]
                x2222 = line4[1][0]
                y2222 = line4[1][1]
                # cx2 = int((x11 + x22)/2)
                # cy2 = int((y11 + y22)/2)
                # lane_color2 = image_ori[cy2,cx2]
                # print("lane_color2",lane_color2)
                cv2.line(img,(x1111,y1111),(x2222,y2222),color4,3)

                class11_5_line = class11_12_12[4]
                line5 = calcu_area_line(class11_5_line)
                x11111 = line5[0][0]
                y11111 = line5[0][1]
                x22222 = line5[1][0]
                y22222 = line5[1][1]
                cv2.line(img,(x11111,y11111),(x22222,y22222),color4,3)

                class11_6_line = class11_12_12[5]
                line6 = calcu_area_line(class11_6_line)
                x111111 = line6[0][0]
                y111111 = line6[0][1]
                x222222 = line6[1][0]
                y222222 = line6[1][1]
                cv2.line(img,(x111111,y111111),(x222222,y222222),color4,3)

                class11_7_line = class11_12_12[5]
                line6 = calcu_area_line(class11_6_line)
                x111111 = line6[0][0]
                y111111 = line6[0][1]
                x222222 = line6[1][0]
                y222222 = line6[1][1]
                cv2.line(img,(x111111,y111111),(x222222,y222222),color4,3)


                line12 = [line1, line2, line3,line4, line5, line6, line7]
                
                return line12

        # if class11_12[0] == [] :
        #     class11_2_line = class11_12_12[1]
        #     line2 = calcu_area_line(class11_2_line)
        #     x11 = line2[0][0]
        #     y11 = line2[0][1]
        #     x22 = line2[1][0]
        #     y22 = line2[1][1]
        #     cv2.line(img,(x11,y11),(x22,y22),color2,3)
        #     class11_3_line = class11_12_12[2]
        #     line3 = calcu_area_line(class11_3_line)
        #     x111 = line3[0][0]
        #     y111 = line3[0][1]
        #     x222 = line3[1][0]
        #     y222 = line3[1][1]
        #     cv2.line(img,(x111,y111),(x222,y222),color2,3)
        #     line12 = [line2, line3]
        #     return line12

        # if class11_12[1] == [] :
        #     class11_1_line = class11_12_12[0]
        #     line1 = calcu_area_line(class11_1_line)
        #     # print("line1",line1)
        #     x1 = line1[0][0]
        #     y1 = line1[0][1]
        #     x2 = line1[1][0]
        #     y2 = line1[1][1]
        #     # cx1 = int((x1 + x2)/2)
        #     # cy1 = int((y1 + y2)/2)
        #     # lane_color1 = image_ori[cy1,cx1]
        #     # print("lane_color1",lane_color1)
        #     cv2.line(img,(x1,y1),(x2,y2),color1,3)
        #     class11_3_line = class11_12_12[2]
        #     line3 = calcu_area_line(class11_3_line)
        #     x111 = line3[0][0]
        #     y111 = line3[0][1]
        #     x222 = line3[1][0]
        #     y222 = line3[1][1]
        #     cv2.line(img,(x111,y111),(x222,y222),color2,3)
        #     line12 = [line1, line3]
        #     return line12




    else:
        
        class11_12 = [class11_line]
        class11_1_line = class11_12[0]
        line1 = calcu_area_line(class11_1_line)
        x1 = line1[0][0]
        y1 = line1[0][1]
        x2 = line1[1][0]
        y2 = line1[1][1]
        # cx1 = int((x1 + x2)/2)
        # cy1 = int((y1 + y2)/2)
        # lane_color1 = image_ori[cy1,cx1]
        # print("lane_color1",lane_color1)
        cv2.line(img,(x1,y1),(x2,y2),color1,3)
        # print("asd",class11_12)
        line12 = [line1]
        # for x1,y1,x2,y2 in class11_1_line:
        #     cv2.line(img,(x1,y1),(x2,y2),color1,1)
        return line12




# def divide_4_lines(sum_roi, d_ab, k, class11_line)::
#     if len(class11_line) == 2:
#         line12_1 = divide_paralle_line(sum_roi, d_ab, k, class11_line):
#         if len(line12_1) = 1:
#             line1 = line12_1[0]
#             line12_2 = divide_coaxis_line(class11_line[1],img,edge,color3,color4,image_ori)
#             if len(line12_2) = 1:
#                 line2 = line12_2[0]
#                 line = [line1, line2]
#                 return line
#             if len(line12_2) = 2:
#                 line2 = line12_2[0]
#                 line3 = line12_2[1]
#                 line = [line1, line2, line3]    

#         if len(line12_1) = 2:
#             line1 = line12_1[0]
#             line2 = line12_1[1]
#             line12_2 = divide_coaxis_line(class11_line[1],img,edge,color3,color4,image_ori)
#             if len(line12_2) = 1:
#                 line3 = line12_2[0]
#                 line = [line1, line2, line3]
#                 return line
#             if len(line12_2) = 2:
#                 line3 = line12_2[0]
#                 line4 = line12_2[1]
#                 line = [line1, line2, line3, line4]
#                 return line        
        




def draw_all_line(line_point11_12,image_ori,point_color):

    if line_point11_12 == None:
        return image_ori
    if len(line_point11_12) == 1:
            
        line_point11_1 = line_point11_12[0]
        # if (line_point11_1[0] == []):
        #     continue
        # x_begin_value1 = line_point11_1[0]
        # x_end_value1 = line_point11_1[2]
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)
        # print("class11_12",class11_12)
        return image_ori

    if len(line_point11_12) == 2:
        
        line_point11_1 = line_point11_12[0]
        # if (line_point11_1[0] == []):
        #     continue
        # x_begin_value1 = line_point11_1[0]
        # x_end_value1 = line_point11_1[2]
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)

        line_point11_2 = line_point11_12[1]
        # if (line_point11_2[0] == []):
        #     continue
        # x_begin_value1 = line_point11_1[0]
        # x_end_value1 = line_point11_1[2]
        y_begin_copy2= copy.deepcopy(line_point11_2[1])
        y_end_copy2 = copy.deepcopy(line_point11_2[3])
        line_begin_coe2, y_begin_value2, x_begin_value_new2 = line_fit_show_y(line_point11_2[0], y_begin_copy2)
        line_end_coe2, y_end_value2, x_end_value_new2 = line_fit_show_y(line_point11_2[2], y_end_copy2)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new2, y_begin_value2, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new2, y_end_value2, point_color)


    if len(line_point11_12) == 3:
        
        line_point11_1 = line_point11_12[0]
     
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)

        line_point11_2 = line_point11_12[1]
       
        y_begin_copy2= copy.deepcopy(line_point11_2[1])
        y_end_copy2 = copy.deepcopy(line_point11_2[3])
        line_begin_coe2, y_begin_value2, x_begin_value_new2 = line_fit_show_y(line_point11_2[0], y_begin_copy2)
        line_end_coe2, y_end_value2, x_end_value_new2 = line_fit_show_y(line_point11_2[2], y_end_copy2)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new2, y_begin_value2, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new2, y_end_value2, point_color)

        line_point11_3 = line_point11_12[2]
       
        y_begin_copy3= copy.deepcopy(line_point11_3[1])
        y_end_copy3 = copy.deepcopy(line_point11_3[3])
        line_begin_coe3, y_begin_value3, x_begin_value_new3 = line_fit_show_y(line_point11_3[0], y_begin_copy3)
        line_end_coe3, y_end_value3, x_end_value_new3 = line_fit_show_y(line_point11_3[2], y_end_copy3)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new3, y_begin_value3, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new3, y_end_value3, point_color)


    if len(line_point11_12) == 4:
        
        line_point11_1 = line_point11_12[0]
    
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)

        line_point11_2 = line_point11_12[1]
    
        y_begin_copy2= copy.deepcopy(line_point11_2[1])
        y_end_copy2 = copy.deepcopy(line_point11_2[3])
        line_begin_coe2, y_begin_value2, x_begin_value_new2 = line_fit_show_y(line_point11_2[0], y_begin_copy2)
        line_end_coe2, y_end_value2, x_end_value_new2 = line_fit_show_y(line_point11_2[2], y_end_copy2)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new2, y_begin_value2, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new2, y_end_value2, point_color)

        line_point11_3 = line_point11_12[2]
    
        y_begin_copy3= copy.deepcopy(line_point11_3[1])
        y_end_copy3 = copy.deepcopy(line_point11_3[3])
        line_begin_coe3, y_begin_value3, x_begin_value_new3 = line_fit_show_y(line_point11_3[0], y_begin_copy3)
        line_end_coe3, y_end_value3, x_end_value_new3 = line_fit_show_y(line_point11_3[2], y_end_copy3)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new3, y_begin_value3, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new3, y_end_value3, point_color)

        line_point11_4 = line_point11_12[3]
    
        y_begin_copy4= copy.deepcopy(line_point11_4[1])
        y_end_copy4 = copy.deepcopy(line_point11_4[3])
        line_begin_coe4, y_begin_value4, x_begin_value_new4 = line_fit_show_y(line_point11_4[0], y_begin_copy4)
        line_end_coe4, y_end_value4, x_end_value_new4 = line_fit_show_y(line_point11_4[2], y_end_copy4)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new4, y_begin_value4, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new4, y_end_value4, point_color)

    if len(line_point11_12) == 5:
        
        line_point11_1 = line_point11_12[0]
    
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)

        line_point11_2 = line_point11_12[1]
    
        y_begin_copy2= copy.deepcopy(line_point11_2[1])
        y_end_copy2 = copy.deepcopy(line_point11_2[3])
        line_begin_coe2, y_begin_value2, x_begin_value_new2 = line_fit_show_y(line_point11_2[0], y_begin_copy2)
        line_end_coe2, y_end_value2, x_end_value_new2 = line_fit_show_y(line_point11_2[2], y_end_copy2)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new2, y_begin_value2, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new2, y_end_value2, point_color)

        line_point11_3 = line_point11_12[2]
    
        y_begin_copy3= copy.deepcopy(line_point11_3[1])
        y_end_copy3 = copy.deepcopy(line_point11_3[3])
        line_begin_coe3, y_begin_value3, x_begin_value_new3 = line_fit_show_y(line_point11_3[0], y_begin_copy3)
        line_end_coe3, y_end_value3, x_end_value_new3 = line_fit_show_y(line_point11_3[2], y_end_copy3)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new3, y_begin_value3, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new3, y_end_value3, point_color)

        line_point11_4 = line_point11_12[3]
    
        y_begin_copy4= copy.deepcopy(line_point11_4[1])
        y_end_copy4 = copy.deepcopy(line_point11_4[3])
        line_begin_coe4, y_begin_value4, x_begin_value_new4 = line_fit_show_y(line_point11_4[0], y_begin_copy4)
        line_end_coe4, y_end_value4, x_end_value_new4 = line_fit_show_y(line_point11_4[2], y_end_copy4)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new4, y_begin_value4, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new4, y_end_value4, point_color)

        line_point11_5 = line_point11_12[4]
    
        y_begin_copy5= copy.deepcopy(line_point11_5[1])
        y_end_copy5 = copy.deepcopy(line_point11_5[3])
        line_begin_coe5, y_begin_value5, x_begin_value_new5 = line_fit_show_y(line_point11_5[0], y_begin_copy5)
        line_end_coe5, y_end_value5, x_end_value_new5 = line_fit_show_y(line_point11_5[2], y_end_copy5)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new5, y_begin_value5, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new5, y_end_value5, point_color)

    if len(line_point11_12) == 6:
            
        line_point11_1 = line_point11_12[0]
    
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)

        line_point11_2 = line_point11_12[1]
    
        y_begin_copy2= copy.deepcopy(line_point11_2[1])
        y_end_copy2 = copy.deepcopy(line_point11_2[3])
        line_begin_coe2, y_begin_value2, x_begin_value_new2 = line_fit_show_y(line_point11_2[0], y_begin_copy2)
        line_end_coe2, y_end_value2, x_end_value_new2 = line_fit_show_y(line_point11_2[2], y_end_copy2)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new2, y_begin_value2, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new2, y_end_value2, point_color)

        line_point11_3 = line_point11_12[2]
    
        y_begin_copy3= copy.deepcopy(line_point11_3[1])
        y_end_copy3 = copy.deepcopy(line_point11_3[3])
        line_begin_coe3, y_begin_value3, x_begin_value_new3 = line_fit_show_y(line_point11_3[0], y_begin_copy3)
        line_end_coe3, y_end_value3, x_end_value_new3 = line_fit_show_y(line_point11_3[2], y_end_copy3)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new3, y_begin_value3, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new3, y_end_value3, point_color)

        line_point11_4 = line_point11_12[3]
    
        y_begin_copy4= copy.deepcopy(line_point11_4[1])
        y_end_copy4 = copy.deepcopy(line_point11_4[3])
        line_begin_coe4, y_begin_value4, x_begin_value_new4 = line_fit_show_y(line_point11_4[0], y_begin_copy4)
        line_end_coe4, y_end_value4, x_end_value_new4 = line_fit_show_y(line_point11_4[2], y_end_copy4)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new4, y_begin_value4, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new4, y_end_value4, point_color)

        line_point11_5 = line_point11_12[4]
    
        y_begin_copy5= copy.deepcopy(line_point11_5[1])
        y_end_copy5 = copy.deepcopy(line_point11_5[3])
        line_begin_coe5, y_begin_value5, x_begin_value_new5 = line_fit_show_y(line_point11_5[0], y_begin_copy5)
        line_end_coe5, y_end_value5, x_end_value_new5 = line_fit_show_y(line_point11_5[2], y_end_copy5)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new5, y_begin_value5, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new5, y_end_value5, point_color)

        line_point11_6 = line_point11_12[5]

        y_begin_copy6= copy.deepcopy(line_point11_6[1])
        y_end_copy6 = copy.deepcopy(line_point11_6[3])
        line_begin_coe6, y_begin_value6, x_begin_value_new6 = line_fit_show_y(line_point11_6[0], y_begin_copy6)
        line_end_coe6, y_end_value6, x_end_value_new6 = line_fit_show_y(line_point11_6[2], y_end_copy6)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new6, y_begin_value6, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new6, y_end_value6, point_color)

    if len(line_point11_12) == 7:
            
        line_point11_1 = line_point11_12[0]
    
        y_begin_copy1= copy.deepcopy(line_point11_1[1])
        y_end_copy1 = copy.deepcopy(line_point11_1[3])
        line_begin_coe1, y_begin_value1, x_begin_value_new1 = line_fit_show_y(line_point11_1[0], y_begin_copy1)
        line_end_coe1, y_end_value1, x_end_value_new1 = line_fit_show_y(line_point11_1[2], y_end_copy1)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new1, y_begin_value1, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new1, y_end_value1, point_color)

        line_point11_2 = line_point11_12[1]
    
        y_begin_copy2= copy.deepcopy(line_point11_2[1])
        y_end_copy2 = copy.deepcopy(line_point11_2[3])
        line_begin_coe2, y_begin_value2, x_begin_value_new2 = line_fit_show_y(line_point11_2[0], y_begin_copy2)
        line_end_coe2, y_end_value2, x_end_value_new2 = line_fit_show_y(line_point11_2[2], y_end_copy2)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new2, y_begin_value2, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new2, y_end_value2, point_color)

        line_point11_3 = line_point11_12[2]
    
        y_begin_copy3= copy.deepcopy(line_point11_3[1])
        y_end_copy3 = copy.deepcopy(line_point11_3[3])
        line_begin_coe3, y_begin_value3, x_begin_value_new3 = line_fit_show_y(line_point11_3[0], y_begin_copy3)
        line_end_coe3, y_end_value3, x_end_value_new3 = line_fit_show_y(line_point11_3[2], y_end_copy3)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new3, y_begin_value3, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new3, y_end_value3, point_color)

        line_point11_4 = line_point11_12[3]
    
        y_begin_copy4= copy.deepcopy(line_point11_4[1])
        y_end_copy4 = copy.deepcopy(line_point11_4[3])
        line_begin_coe4, y_begin_value4, x_begin_value_new4 = line_fit_show_y(line_point11_4[0], y_begin_copy4)
        line_end_coe4, y_end_value4, x_end_value_new4 = line_fit_show_y(line_point11_4[2], y_end_copy4)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new4, y_begin_value4, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new4, y_end_value4, point_color)

        line_point11_5 = line_point11_12[4]
    
        y_begin_copy5= copy.deepcopy(line_point11_5[1])
        y_end_copy5 = copy.deepcopy(line_point11_5[3])
        line_begin_coe5, y_begin_value5, x_begin_value_new5 = line_fit_show_y(line_point11_5[0], y_begin_copy5)
        line_end_coe5, y_end_value5, x_end_value_new5 = line_fit_show_y(line_point11_5[2], y_end_copy5)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new5, y_begin_value5, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new5, y_end_value5, point_color)

        line_point11_6 = line_point11_12[5]

        y_begin_copy6= copy.deepcopy(line_point11_6[1])
        y_end_copy6 = copy.deepcopy(line_point11_6[3])
        line_begin_coe6, y_begin_value6, x_begin_value_new6 = line_fit_show_y(line_point11_6[0], y_begin_copy6)
        line_end_coe6, y_end_value6, x_end_value_new6 = line_fit_show_y(line_point11_6[2], y_end_copy6)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new6, y_begin_value6, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new6, y_end_value6, point_color)

        line_point11_7 = line_point11_12[6]

        y_begin_copy7= copy.deepcopy(line_point11_7[1])
        y_end_copy7 = copy.deepcopy(line_point11_7[3])
        line_begin_coe7, y_begin_value7, x_begin_value_new7 = line_fit_show_y(line_point11_7[0], y_begin_copy7)
        line_end_coe7, y_end_value7, x_end_value_new7 = line_fit_show_y(line_point11_7[2], y_end_copy7)
        image_ori = draw_line_in_image(image_ori, x_begin_value_new7, y_begin_value7, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new7, y_end_value7, point_color)

    return image_ori

def line_detection(edge,image_ori,point_color,line_flag):
    # point_color = color_table()[1]
    lines = cv2.HoughLinesP(edge,1,np.pi/360,50,minLineLength=45,maxLineGap=35)
    if lines is not None:
        
        lines1 = lines[:,0,:]#提取为二维
        
            
        # print(lines1)
        line_selected = [] 
        D = []
        vecs = []
        vecs_inv = []
        lines_new = []
        img = np.zeros((600,480,3),np.uint8)
        img[:,:,0] = edge
        img[:,:,1] = edge
        img[:,:,2] = edge
        for x1,y1,x2,y2 in lines1[:]: 
            d = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            if line_flag == center_lane_flag:
                distance = 80
            else:
                distance = 80
            if d >= distance:
                D.append(d)
            
                vecs.append([(x2-x1)/d,(y2-y1)/d])
                
                lines_new.append([x1,y1,x2,y2])
        
                # vecs_inv.append([(y2-y1)/d,(x2-x1)/d])
        lines_new = np.array(lines_new)
        if len(lines_new) != 0:
            # exit()
            vecs = np.array(vecs)
            # print(lines_new)
            vec_pro = []
            vec1 = vecs[0]
            # vec2 = vecs[1]
            # print("vec1",vec1[0]*vec2[0] + vec1[1]*vec2[1])
            for vec in vecs:
                vec_pro.append(abs(vec1[0]*vec[0] + vec1[1]*vec[1]))
            # print("vec_pro",vec_pro)
            # the class1 lines
            class1_index = np.where(np.array(vec_pro) >= 0.85)[0]
            class1_line = lines_new[class1_index]
            class1_vector = vecs[class1_index]

            class1_12 = divide_by_distance(class1_line,class1_vector)
            # print("class1_12",class1_12)
            # line_point_all = []

            if len(class1_12) == 1:
                class11_line = class1_12[0]
                class11_12 = divide_coaxis_line(class11_line,img,edge,(0,255,0),(255,0,0),(0,255,0),(255,0,0),image_ori)
                line_point11_12 = get_fit_point_1_or_2(edge,class11_12)
                image_ori = draw_all_line(line_point11_12,image_ori,point_color)
                # line_point_all.append(line_point11_12)

                
            if len(class1_12) == 2:
                class11_line = class1_12[0]
                class11_12 = divide_coaxis_line(class11_line,img,edge,(0,255,0),(255,0,0),(0,255,0),(255,0,0),image_ori)
                # print("class11_12",class11_12)
                line_point11_12 = get_fit_point_1_or_2(edge,class11_12)
                image_ori = draw_all_line(line_point11_12,image_ori,point_color)
                # line_point_all.append(line_point11_12)
                # print("***************")
                class12_line = class1_12[1]
                class12_12 = divide_coaxis_line(class12_line,img,edge,(0,0,255),(255,255,0),(0,0,255),(255,255,0),image_ori)
                line_point12_12 = get_fit_point_1_or_2(edge,class12_12)
                image_ori = draw_all_line(line_point12_12,image_ori,point_color)
                # line_point_all.append(line_point12_12)
            # print("*****************")
            # print("class1 index:", class1_index) 
            # print("class1_line:",class1_line)
            # print("number if class1:",len(class1_index))


            # the class2 lines
            if len(class1_index) < len(vec_pro):
                # print("***********************")
                # print("there are 2 classes of the road lanes")
                # print("***********************")
                class2_index = np.where(np.array(vec_pro) <= 0.6)[0]
                class2_line = lines_new[class2_index]
                class2_vector = vecs[class2_index]
                class2_12 = divide_by_distance(class2_line,class2_vector)
                # print("class2_12",class2_12)
                if len(class2_12) == 1:
                    class21_line = class2_12[0]
                    # print(class21_line)
                    class21_12 = divide_coaxis_line(class21_line,img,edge,(0,255,255),(255,0,255),(0,255,255),(255,0,255),image_ori)
                    line_point21_12 = get_fit_point_1_or_2(edge,class21_12)
                    image_ori = draw_all_line(line_point21_12,image_ori,point_color)
                    # line_point_all.append(line_point21_12)
                
                if len(class2_12) == 2:
                    class21_line = class2_12[0]
                    class21_12 = divide_coaxis_line(class21_line,img,edge,(0,255,255),(255,0,255),(0,255,255),(255,0,255),image_ori)
                    line_point21_12 = get_fit_point_1_or_2(edge,class21_12)
                    image_ori = draw_all_line(line_point21_12,image_ori,point_color)
                    # line_point_all.append(line_point21_12)

                    class22_line = class2_12[1]
                    # print("22",class21_line)
                    class22_12 = divide_coaxis_line(class22_line,img,edge,(223,133,170),(100,20,20),(223,133,170),(100,20,20),image_ori)
                    line_point22_12 = get_fit_point_1_or_2(edge,class22_12)
                    image_ori = draw_all_line(line_point22_12,image_ori,point_color)
                # line_point_all.append(line_point22_12)
            
            # print("*****************")
            # print("class2 index:", class2_index) 
            # print("class2_line:",class2_line)
            # print("number if class2:",len(class2_index))
        # print("***************")
        # print("vec_pro",vec_pro)
        # print(len(lines1))
        # print(D)
        # print(len(D))
        # print(vecs)
        cv2.imwrite( "../lane_fitting/20200403_shoukai/edge.jpg",img)
        # cv2.imwrite( "../lane_fitting/20200403_shoukai/image_ori.jpg",image_ori)
        # cv.imshow("line_detection", edge)
    return image_ori



def color_table():
    color_list = [[0, 255, 0], [0, 10, 255], [255, 0, 0]]
    return color_list

def divide_2parts(test_index):
    
    mid1 = np.where(test_index[1] <= 240)[0]
    mid2 = np.where(test_index[1] > 240)[0]
    # mid = test_index[1][Mid]
    # print(mid1)
    # print(mid2)
    x_test1 = test_index[1][mid1,np.newaxis]
    y_test1 = test_index[0][mid1,np.newaxis]
    x_test2 = test_index[1][mid2,np.newaxis]
    y_test2 = test_index[0][mid2,np.newaxis]

    X1=np.hstack((x_test1,y_test1))
    X2=np.hstack((x_test2,y_test2))

    return [X1,X2]

# def calcu_rough_direction(X):
#     cov = np.cov(X.T)
#     print(cov)
#     eigen_vals,eigen_vecs = np.linalg.eig(cov)
#     where_max_eigen_vals = np.where( eigen_vals == max(eigen_vals))[0]
#     print(where_max_eigen_vals)
#     vector_max = eigen_vecs[:,where_max_eigen_vals]
#     k = vector_max[1]/vector_max[0]
#     print("test_line_cov",cov)
#     print("eigen_vals",eigen_vals)
#     print("eigen_vecs",eigen_vecs)
#     print("slope_k:",k)
#     return k


def draw_line_according_to_cls(image_one, line_flag, image_ori, point_color):
    height, width = image_one.shape
    # print(height,width)
    # 1.class group
    # np.where(condition,x,y) 满足条件输出x，不满足条件输出y，此处，将两侧车道线置为1，其余位置置为0
    line_1 = np.where(image_one == line_flag, 255.0, 0.0).astype('uint8')
    image_line_flag = line_1
    # image_line_flag.astype('uint8')
    # image_line_flag.dtype=np.uint8
    # print(image_line_flag)
    # print(image_line_flag.shape)

    image_ori = line_detection(image_line_flag,image_ori,point_color,line_flag)
    return image_ori
    # print("all the lines:",line_all)
    
    # plt.imshow(image_line_flag)
    # test_line = line_1[:150,:]
    # print(test_line)
    
    # test_index = np.where(test_line == 1)
    # # mid = np.where(test_index[1] <= 240)[0]
    # # print("mid",mid)
    # print("test_index",test_index)
    # divide = divide_2parts(test_index)
    # X1 = divide[0]
    # # print("x1",X1)
    # print("number of x1",len(X1))
    # X2 = divide[1]
    # # print("x2",X2)
    # print("number of x2",len(X2))
    # if len(X1)>5:
    #     k1 = calcu_rough_direction(X1)

    # # print(X2)
    # if len(X2)>5:
    #     k2 = calcu_rough_direction(X2)




if __name__ == '__main__':
    # root_path = "C:/Users/man.wang/Desktop/ASP/20200309_asp_small_model_demo/"
    # print(float("inf")>100)
   

    root_path = "../lane_fitting/20200403_shoukai/"
    for demo_list in os.listdir(root_path):
        print('********************* fit {} ********************'.format(demo_list))
        image_ori_path = root_path + demo_list + '/img/'
        test_result_path = root_path+ demo_list + '/test_result/'
        image_path_save = root_path+ demo_list + '/fit_result_demo/'
        mkdir(image_path_save)
        for i in os.listdir(test_result_path):
                # if i == "000707.png":
                    image_name = i
                    #print("----image_name----",image_name)
                    #image_name = '0321.png'
                    test_result_path_all = test_result_path + image_name
                    image = cv2.imread(test_result_path_all)
                    image_ori_name = image_name.split(".png")[0] + ".jpg"
                    print('image:'+image_ori_name)
                    image_ori_path_all = image_ori_path + image_ori_name
                    image_ori = cv2.imread(image_ori_path_all)
                    high, width,c = image.shape
                    image_one = image[:,:,0]
                    print(image_one.shape)
                    # draw lane
                    print("lane")
                    # print("***********************")
                    point_color = color_table()[1]
                    image_ori = draw_line_according_to_cls(image_one,line_2_flag,image_ori,point_color)

                    # # draw parking slot lane
                    print("slot lane")
                    print("**********************")
                    point_color = color_table()[2]
                    image_ori = draw_line_according_to_cls(image_one, parking_slot_lane_flag, image_ori, point_color)

                    #draw center lane
                    print("center lane")
                    print("**********************")
                    point_color = color_table()[0]
                    image_ori = draw_line_according_to_cls(image_one,center_lane_flag,image_ori,point_color)

                    image_save_name = image_path_save + image_name
                    cv2.imwrite(image_save_name,image_ori)


    # 1。根据方向，两个方向向量积分类
    # 2。一个方向的线段类，任意一点，我去第一个点也算随机了，然后算点与其他同类直线的最短距离，本俯视图中车道线像素一般为20左右，大概在在这值左右设立一个阈值，距离太远的就是对面的线
    # 3。 在三岔路口和十字路口，所以粗略想了去第二次分类后的直线会被断掉，所以去所有直线集合里所有的（X1,Y1) (X2,Y2)算最大距离，去坐标算中的，然后根据斜率进行切割分类
    