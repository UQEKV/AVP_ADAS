# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import copy

# https://www.cnblogs.com/jingsupo/p/python_curve_fit.html
# line_1_flag = 1
parking_slot_lane_flag = 1
line_2_flag = 2
center_lane_flag = 8
mode = 'TJP'


def mkdir(path):
    if (os.path.exists(path) == False):
        os.mkdir(path)
    else:
        pass


def filter_error_record(index_record, line_index):
    threshold = 10
    index_record_new = []
    if (line_index != []):
        index_record_new.append(line_index[0])
    else:
        return index_record_new

    for i in range(len(index_record)):
        sub_before = abs(line_index[index_record[i]] - line_index[index_record[i] - 1])
        if (index_record[i] + 1 < len(line_index)):
            sub_after = abs(line_index[index_record[i] + 1] - line_index[index_record[i]])
        else:
            pass
        if (sub_before > threshold and sub_after > threshold):
            pass
        else:
            index_record_new.append(line_index[index_record[i]])
    index_record_new.append(line_index[len(line_index) - 1])
    # if len(index_record_new) % 2 != 0:
    #     if index_record_new[0] == line_index[0]:
    #         index_record_new.remove(index_record_new[0])
    #     elif index_record_new[-1] == line_index[-1]:
    #         index_record_new.remove(index_record_new[-1])
    return index_record_new


def filter_error_detection_according_x_width(index_record):
    x_width_thresh = 5
    index_record_copy = copy.deepcopy(index_record)
    for i in range(len(index_record) / 2):
        sub_x = index_record[2 * i + 1] - index_record[2 * i]
        if (sub_x < x_width_thresh):
            index_record_copy.remove(index_record[2 * i + 1])
            index_record_copy.remove(index_record[2 * i])
    return index_record_copy


def filter_point(x, y):
    x_new = []
    y_new = []
    step = 2
    for i in range(len(x)):
        if (i % step == 0):
            x_new.append(x[i])
            y_new.append(y[i])
    return x_new, y_new


def line_fit_show_y(x, y, n=3):
    # z1 = np.polyfit(x, y, n) # 用3次多项式拟合
    # z1 是一个向量，包含四个值，表示(ax^3+bx^2+cx+d)
    z1 = np.polyfit(y, x, n)  # 用3次多项式拟合
    # 将系数带入方程，得到一个函数式子 p1
    p1 = np.poly1d(z1)
    # xvals = get_fit_x_according_y(y,p1,x)
    # print(p1) # 在屏幕上打印拟合多项式
    xvals = p1(y)  # 也可以使用yvals=np.polyval(z1,x)

    return p1, y, xvals


def get_fit_point(image_line_flag, line_index, x_begin, x_end):
    high, width = image_line_flag.shape
    row_start = 0
    step_row = 10
    step_col = 1
    value_flag = 1
    x_line_begin = x_begin - 1
    x_line_end = x_end + 1
    # x_begin = x_begin + 1
    # x_end = x_end - 1
    if (x_line_begin <= 0):
        x_line_begin = 1
    image_width = 480
    if (x_line_end >= image_width):
        x_line_end = image_width - 1

    row_end = high - row_start
    begin_point_x = []
    begin_point_y = []
    for row in range(row_start, row_end, step_row):
        for col in range(x_line_begin, x_line_end, step_col):
            if (image_line_flag[row, col] == value_flag):
                # point = [row,col]
                # begin_point.append(point)
                begin_point_x.append(col)
                begin_point_y.append(row)
                break
    end_point_x = []
    end_point_y = []
    for row in range(row_start, row_end, step_row):
        for col in range(x_line_end, x_line_begin, -step_col):
            if (image_line_flag[row, col] == value_flag):
                # point = [row,col]
                # end_point.append(point)
                end_point_x.append(col)
                end_point_y.append(row)
                break
    return [begin_point_x, begin_point_y, end_point_x, end_point_y]


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


def filter_error_according_pixel_sum(index_record, image_one):
    # sum_threshold = 150
    sum_threshold = 300
    index_record_copy = copy.deepcopy(index_record)
    data_len = int(len(index_record) * 0.5)
    for i in range(data_len):
        begin = index_record[2 * i + 0]
        end = index_record[2 * i + 1]
        image_sum = np.sum(image_one[:, begin:end])
        if (image_sum < sum_threshold):
            index_record_copy.remove(begin)
            index_record_copy.remove(end)
    return index_record_copy


def divide_line_side(line_num, index_record, image_line_flag, image_ori, line_index, point_color):
    # loop for all line
    for line_num_one in range(line_num):
        x_begin = index_record[line_num_one * 2 + 0]
        x_end = index_record[line_num_one * 2 + 1]
        line_point = get_fit_point(image_line_flag, line_index, x_begin, x_end)
        # decide null point fit
        if (line_point[0] == []):
            continue
        x_begin_value = line_point[0]
        x_end_value = line_point[2]
        y_begin_copy = copy.deepcopy(line_point[1])
        y_end_copy = copy.deepcopy(line_point[3])
        line_begin_coe, y_begin_value, x_begin_value_new = line_fit_show_y(line_point[0], y_begin_copy)
        line_end_coe, y_end_value, x_end_value_new = line_fit_show_y(line_point[2], y_end_copy)

        image_ori = draw_line_in_image(image_ori, x_begin_value_new, y_begin_value, point_color)
        image_ori = draw_line_in_image(image_ori, x_end_value_new, y_end_value, point_color)
    return image_ori


def merge_index(line_index):   #find where is the other lane
    index_record = []
    # merge index
    merge_thresh = 4
    for index, value in enumerate(line_index):
        # 遍历 line_index 这个向量内的所有值，当索引值（index）大于向量长度时，跳出循环
        if (index >= line_index.shape[0] - 1):
            break
        sub_value = line_index[index + 1] - line_index[index]
        # if(sub_value!=1):
        if (sub_value >= merge_thresh):
            index_record.append(index)
            index_record.append(index + 1)
    return index_record


def draw_line_according_to_cls(image_one, line_flag, image_ori, point_color):
    height, width = image_one.shape
    print(height,width)
    # 1.class group
    # np.where(condition,x,y) 满足条件输出x，不满足条件输出y，此处，将两侧车道线置为1，其余位置置为0
    line_1 = np.where(image_one == line_flag, 1, 0)
    image_line_flag = line_1
    # del horizontal pixel from horizontal projection
    # col_threshold = 35
    if mode == 'ASP':
        # print('------------>ASP')
        col_threshold = 30
        col_sum = np.zeros(height)
        for col in range(height):
            col_sum[col] = np.sum(line_1[col, :])
        col_index = np.where(col_sum > col_threshold)
        print('---col_index----shape:',col_index.shape)
        print('---col_index----:',col_index)
        col_index = col_index[0]
        print('---col_index----shape:',col_index.shape)
        print('---col_index----:',col_index)
        for i in range(len(col_index)):
            col_i = col_index[i]
            image_line_flag[col_i, :] = 0
        line_1 = image_line_flag
        print('---line_1----shape:',line_1.shape)
        print('---line_1----:',line_1)
    # divide ground  projection
    row_sum = np.zeros(width)
    for row in range(width):  # 纵向求和，得到有值的列
        row_sum[row] = np.sum(line_1[:, row])
        
    print('------row____shape',row_sum.shape)
    print('------row____',row_sum)
    line_index = np.where(row_sum > 0)
    print('---line_index----:',line_index)
    
    line_index = line_index[0]
    print('---line_index----:',line_index)
    print('---line_index----shape:',line_index.shape)
    # if len(line_index)>2:
    #     while True:
    #         if line_index[1]-line_index[0]>1:
    #             line_index = np.delete(line_index,0)
    #         elif line_index[-1]-line_index[-2]>1:
    #             line_index = np.delete(line_index,-1)
    #         else:
    #             break
    while True:
        if len(line_index) > 2:
            if line_index[1] - line_index[0] > 1:
                line_index = np.delete(line_index, 0)
            elif line_index[-1] - line_index[-2] > 1:
                line_index = np.delete(line_index, -1)
            else:
                break
        else:
            break
    print('---line_index----:',line_index)
    # merge index
    index_record = merge_index(line_index)
    print("index_record",index_record)

    # filter error index
    index_record = filter_error_record(index_record, line_index)
    index_record = filter_error_detection_according_x_width(index_record)
    # index_record = filter_error_according_pixel_sum(index_record,image_one)
    index_record = filter_error_according_pixel_sum(index_record, image_line_flag)
    line_num = int(len(index_record) * 0.5)
    # divide line left and right
    image_ori = divide_line_side(line_num, index_record, image_line_flag, image_ori, line_index, point_color)
    return image_ori
# \\lzding\share_lz\73_ASP\04_demo\20200305_TJP_demo
if __name__ == '__main__':
    # root_path = "C:/Users/man.wang/Desktop/ASP/20200309_asp_small_model_demo/"
    root_path = "../20200403_shoukai/"
    for demo_list in os.listdir(root_path):
        print('********************* fit {} ********************'.format(demo_list))
        image_ori_path = root_path + demo_list + '/img/'
        test_result_path = root_path+ demo_list + '/test_result/'
        image_path_save = root_path+ demo_list + '/fit_result_demo/'
        mkdir(image_path_save)
        for i in os.listdir(test_result_path):
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
            # print(image_one.shape)
            # draw lane
            point_color = color_table()[1]
            image_ori = draw_line_according_to_cls(image_one,line_2_flag,image_ori,point_color)
            #draw center lane
            point_color = color_table()[0]
            image_ori = draw_line_according_to_cls(image_one,center_lane_flag,image_ori,point_color)
            # draw parking slot lane
            point_color = color_table()[2]
            image_ori = draw_line_according_to_cls(image_one, parking_slot_lane_flag, image_ori, point_color)

            ## ******************* 2 cls ***********************
            # point_color = color_table()[0]
            # image_ori = draw_line_according_to_cls(image_one,white_lane_flag,image_ori,point_color)
            # draw parking slot lane
            # point_color = color_table()[1]
            # image_ori = draw_line_according_to_cls(image_one, yellow_lane_flag, image_ori, point_color)

            image_save_name = image_path_save + image_name
            cv2.imwrite(image_save_name,image_ori)

