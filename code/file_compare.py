import os
import shutil

"""
比较两个一级文件夹中的二级文件夹中的文件，若有相同名称的文件则拷贝出来
相比较的文件夹名字必须相同
只支持两级文件夹
"""


# BASE_FOLDER= r'D:\20191025_changan_videos\20191025_changan_videos_5frame\capture_0' # 未标注数据集所在位置
# COMPARE_FOLDER = r'D:\20191025_changan_videos\20191025_changan_videos_check_2\capture_0_complete' # 挑选出的有错误的数据集所在位置
# DEST_DIR = r'D:\20191025_changan_videos\picked' # 储存位置


def get_dirnames(base_dir):
    """

    :param base_dir:
    :return:
    """
    everythin_in_folder = os.listdir(base_dir)

    all_dirs = map(lambda x: os.path.join(base_dir, x), everythin_in_folder)
    dir_list = list(filter(os.path.isdir, all_dirs))
    return dir_list


def get_filenames(file_folder):
    """

    :param file_folder:
    :return:
    """
    file_list = os.listdir(file_folder)
    file_list = map(lambda x: os.path.join(file_folder, x), file_list)
    file_list = list(filter(os.path.isfile, file_list))
    return file_list


def main(BASE_FOLDER, COMPARE_FOLDER, DEST_DIR):
    folders_for_compare = get_dirnames(BASE_FOLDER)
    print(folders_for_compare)
    folders_to_compare = get_dirnames(COMPARE_FOLDER)
    print(folders_to_compare)
    for folder_to_compare in folders_to_compare:
        if folder_to_compare.split('/')[-1] in list(map(lambda x: x.split('/')[-1], folders_for_compare)):
            folder_for_compare = os.path.join(BASE_FOLDER, folder_to_compare.split('/')[-1])
            files_for_compare = get_filenames(folder_for_compare)
            files_to_compare = get_filenames(folder_to_compare)
            for file_to_compare in files_to_compare:
                if file_to_compare.split('/')[-1] in list(map(lambda x: x.split('/')[-1], files_for_compare)):
                    to_copy_folder = os.path.join(DEST_DIR, folder_to_compare.split('/')[-1])
                    to_copy_file = os.path.join(to_copy_folder, file_to_compare.split('/')[-1])
                    if not os.path.exists(to_copy_folder):
                        os.mkdir(to_copy_folder)
                    file_to_copy = os.path.join(folder_for_compare, file_to_compare.split('/')[-1])
                    shutil.copyfile(file_to_copy, to_copy_file)
                    print("copied from: ", file_to_copy, " to ", to_copy_file)


if __name__ == '__main__':
    BASE_FOLDER = './3th_factory_01_0508_frame/'  # 未标注数据集所在位置
    COMPARE_FOLDER = './1105_output_complete/3th_factory_01_0508_frame_output/'  # 挑选出的有错误的数据集所在位置
    DEST_DIR = './1105_output_selected/3th_factory_01_0508_frame_selected/'  # 储存位置
    # if not os.path.exists(DEST_DIR):
    #                     os.mkdir(DEST_DIR)
    for i in [0]:
        capture_folder = 'capture_' + str(i)
        compare_capture_folder = capture_folder# + '_complete'
        dest_folder = capture_folder
        base_dir = os.path.join(BASE_FOLDER, capture_folder)
        compare_dir = os.path.join(COMPARE_FOLDER, compare_capture_folder)
        dest_dir = os.path.join(DEST_DIR, dest_folder)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        main(base_dir, compare_dir, dest_dir)
