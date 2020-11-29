import cv2
import os

CAPTURE_DIR = './20200323/'
OUTPUT_DIR = './20200323_5frame/'

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
        if (!frame.empty()):
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
            if video.split('/')[-1] != 'rear.avi':
                print('process video: ', video)
                view = video.split('/')[-1].split('.')[0]
                capture_name = capture_folder.split('/')[-1]
                create_output_dir(OUTPUT_DIR, capture_name, view)
                save_dir = get_save_dir(OUTPUT_DIR, capture_name, view)
                extract_frame(video, save_dir, 25)


if __name__ == '__main__':
    main(CAPTURE_DIR, OUTPUT_DIR)

