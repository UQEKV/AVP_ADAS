
import os 
import xml.dom.minidom
import numpy as np
import cv2

def getText(node):
	return node.firstChild.nodeValue

view_output = './view_output'
if not os.path.exists(view_output):
    os.mkdir(view_output)

output_root = './output_image_label/'
output_img = os.path.join(output_root, 'img')
for folder in os.listdir(output_img):
    save_dir = os.path.join(view_output, folder) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_path = os.path.join(output_img, folder) + '/'
    xml_path = os.path.join(output_root, 'xml', folder).replace('extract', 'label') + '/'
    print(save_dir, img_path, xml_path)

    xml_list = os.listdir(xml_path)
    f = os.listdir(img_path)

    for i in range(len(xml_list)):
        xml_name=xml_list[i]
        dom = xml.dom.minidom.parse(xml_path + xml_name)
        root = dom.documentElement
        imgPath = img_path + f[i]
        print("imgPath:",imgPath)
        img = cv2.imread(imgPath)
        for obj in root.getElementsByTagName("object"):
            if getText(obj.getElementsByTagName("name")[0]) == 'car':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (0, 0, 255), 2)
            if getText(obj.getElementsByTagName("name")[0]) == 'rear':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (0, 255, 255), 2)
            if getText(obj.getElementsByTagName("name")[0]) == 'front':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (255, 255, 0), 2)
            if getText(obj.getElementsByTagName("name")[0]) == 'person':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (255, 50, 255), 2)
            if getText(obj.getElementsByTagName("name")[0]) == 'rider':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (255, 150, 255), 2)
            if getText(obj.getElementsByTagName("name")[0]) == 'truck':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (255, 255, 50), 2)
            if getText(obj.getElementsByTagName("name")[0]) == 'bus':
                Xmin = getText(obj.getElementsByTagName("xmin")[0])
                Ymin = getText(obj.getElementsByTagName("ymin")[0])
                Xmax = getText(obj.getElementsByTagName("xmax")[0])
                Ymax = getText(obj.getElementsByTagName("ymax")[0])
                cv2.rectangle(img, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (50, 255, 255), 2)
        savefile = save_dir + f[i].split('.')[0] + '.jpg'
        print('savefile:',savefile)
        cv2.imwrite(savefile, img)
        

