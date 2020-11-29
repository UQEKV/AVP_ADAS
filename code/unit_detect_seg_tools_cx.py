import numpy as np

import timeit
import Image
import ImageDraw 
import os
import glob

from PIL import Image 
import matplotlib.pyplot as plt 
import scipy.misc as sm

import cv2
import types


CAPTURE_DIR = '/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/test_data/temp_10videos/'
OUTPUT_DIR = '/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/test_data/temp_10frame/'
SUB_OUTPUT_DIR_NUMBER = 2   # the number of subfile 
save_dir = '/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/test_data/temp_10frame_output/'  
model_def = './deploy_no_group_320_192.prototxt'
model_weights = './models/modify22_20200421_APA_iter_200000.caffemodel'
voc_labelmap_file = '/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/src/labelmap_voc.prototxt'

index = 0

def get_dirnames(base_dir):
    """

    :param base_dir:
    :return:
    """
    everythin_in_folder = os.listdir(base_dir)

    all_dirs = map(lambda x: os.path.join(base_dir, x), everythin_in_folder)
    dir_list = list(filter(os.path.isdir, all_dirs))
    return dir_list


def postfix_finder(file_folder, postfix):
    """

    :param file_folder:
    :return:
    """
    file_list = os.listdir(file_folder)
    file_list = list(map(lambda x: os.path.join(file_folder, x), file_list))
    file_list = list(filter(lambda x: x.endswith(postfix), file_list))
    return file_list


def create_output_dir(dst_dir, capture, view):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        print(dst_dir, " created")
    if not os.path.exists(os.path.join(dst_dir, capture)):
        os.mkdir(os.path.join(dst_dir, capture))
        print(os.path.join(dst_dir, capture), " created")
    if not os.path.exists(os.path.join(dst_dir, capture, view)):
        os.mkdir(os.path.join(dst_dir, capture, view))
        print(os.path.join(dst_dir, capture, view), " created")


def get_save_dir(dst_dir, capture, view):
    return os.path.join(dst_dir, capture, view)


def extract_frame(video, output_dir, FPS=1):
    videoCapture = cv2.VideoCapture(video)
    i = 0
    while True:
        success, frame = videoCapture.read()
        i += 1
        #if i > 9000 and i < 10000:
        
        if success:
            if (i % FPS == 0):
                global index
                index += 1
                savedname = str(index).zfill(6) + '.jpg'
                output_filename = os.path.join(output_dir, savedname)
                cv2.imwrite(output_filename, frame)
                print('image of %s is saved' % (output_filename))
        if not success:
            print('video is all read')
            break


def main(CAPTURE_DIR, OUTPUT_DIR):
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    capture_folders = get_dirnames((CAPTURE_DIR))
    for capture_folder in capture_folders:
        videos = postfix_finder(capture_folder, 'avi')
        for video in videos:
            # if video.split('/')[-1] != 'rear.avi':    delete the videos from rear camera
            print('process video: ', video)
            view = video.split('/')[-1].split('.')[0]
            capture_name = capture_folder.split('/')[-1]
            create_output_dir(OUTPUT_DIR, capture_name, view)
            save_dir = get_save_dir(OUTPUT_DIR, capture_name, view)
            extract_frame(video, save_dir, 25)


main(CAPTURE_DIR, OUTPUT_DIR)
# if __name__ == '__main__':
#     main(CAPTURE_DIR, OUTPUT_DIR)





plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that the work directory is caffe_root
caffe_root = './' 
Path =  OUTPUT_DIR                #"/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/test_data/temp_10frame/"
img_list= []                                              
# for root,dirs,files in os.walk(Path):
#     for name in files:
#         #print('**********root + name*************',root +'/'+ name)
#         img_list.append(root +'/'+ name)
if SUB_OUTPUT_DIR_NUMBER==2:
    img_list = glob.glob(Path + '*/*/*.jpg')
elif SUB_OUTPUT_DIR_NUMBER==3:
    img_list = glob.glob(Path + '*/*/*/*.jpg')

#save_dir = '/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/test_data/temp_10frame_output/'  


os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import caffe

caffe.set_device(2)
caffe.set_mode_gpu()


# model_def = './deploy_no_group_320_192.prototxt'
# model_weights = './models/modify22_20200421_APA_iter_200000.caffemodel'

# voc_labelmap_file = '/mnt/nfs/zzwu/02_train/rrc_test/rrc/models/wyn_changan_tjp_0327_wheel/Apa_model/src/labelmap_voc.prototxt'

if not(os.path.exists(save_dir)):
    os.makedirs(save_dir)

file = open(voc_labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()

text_format.Merge(str(file.read()), labelmap) 

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1

image_width = 320
image_height = 192

thresh_list = {'car': 0.3, 'person': 0.3, 'truck': 0.3, 'bus': 0.3, 'rider': 0.3, 'rear': 0.3, 'front': 0.3}


#net.blobs['data'].reshape(1,3,image_height,image_width)
    
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

for i in range(len(img_list)):        
    img_file = img_list[i]
    img_info = img_file.split('/')
    img_name = img_info[-1]
    if SUB_OUTPUT_DIR_NUMBER==2:
        file_name = '{}/{}'.format(img_info[-3], img_info[-2])
    elif SUB_OUTPUT_DIR_NUMBER==3:
        file_name = '{}/{}/{}'.format(img_info[-4], img_info[-3], img_info[-2])
    
    
    image = caffe.io.load_image(img_file) 

    transformed_image = transformer.preprocess('data', image)
    transformed_image = transformed_image/255.0
    
    net.blobs['data'].data[...] = transformed_image

    #t1 = timeit.Timer("net.forward()","from __main__ import net")
    #print t1.timeit(2)

    # Forward pass.
    detections = net.forward()['detection_out']
    # print (detections)
    #assert false
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.001
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]   

    #img = Image.open(img_dir + "%06d.jpg"%(img_idx))
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)   

    for i in xrange(top_conf.shape[0]):
        xmin = top_xmin[i] * image.shape[1]
        ymin = top_ymin[i] * image.shape[0]
        xmax = top_xmax[i] * image.shape[1]
        ymax = top_ymax[i] * image.shape[0]

        h = float(ymax - ymin)
        w = float(xmax - xmin)
        #if (w==0) or (h==0):
        #   continue
        #if (h/w >=2)and((xmin<10)or(xmax > 1230)):
        #   continue
        label = top_labels[i]
        score = top_conf[i]
        # thresh = thresh_list[label]
        thresh = 0.2
        
        if score > thresh:
            #print "------------label:",label
            if label =="car":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(0,0,255),width=2)
                #draw.text((xmax,ymax),'%s %.2f'%(label, score),fill=(255,255,255))
            elif label =="person":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(0,255,0),width=2)
                #draw.text((xmin,ymin),'%s %.2f'%(label, score),fill=(255,255,255))
            elif label =="truck":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(255,0,0),width=2)
                #draw.text((xmin,ymin),'%s %.2f'%(label, score),fill=(255,255,255))
            elif label =="bus":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(0,255,255),width=2)
                #draw.text((xmin,ymax),'%s %.2f'%(label, score),fill=(255,255,255))
            elif label =="rider":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(255,255,0),width=2)
                #draw.text((xmin,ymin),'%s %.2f'%(label, score),fill=(255,255,255))
            elif label =="rear":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(255,0,255),width=2)
                #draw.text((xmax,ymin),'%s %.2f'%(label, score),fill=(255,255,255))
            elif label =="front":       
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(68,126,68),width=2)   
                #draw.text((xmin,ymin),'%s %.2f'%(label, score),fill=(255,255,255))
            elif score > 0.2:
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(0,0,0),width=2)
                draw.text((xmin,ymax),'%s %.2f'%(label, score),fill=(255,255,255))                   
    if not(os.path.exists(save_dir +file_name)):
        os.makedirs(save_dir +file_name)
    #if 'truck' in top_labels or 'bus' in top_labels:    
    img.save(save_dir +file_name+'/'+img_name)
     
                                                            
    
