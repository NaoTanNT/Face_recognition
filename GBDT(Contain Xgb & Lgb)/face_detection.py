import numpy as np
import face_recognition
import shutil
import os

# 设置图像文件夹路径
folder_path = "raw_data"

# 创建两个列表，分别保存检测到和没有检测到人脸的文件名
faces_detected_files = []
faces_not_detected_files = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 读取二进制图像文件
    Image_raw = np.fromfile(file_path, dtype=np.uint8)

    # 将二进制数据转换为图片（假设每个图像都是128x128的大小）
    try:
        Image_current = np.reshape(Image_raw, (128, 128))  # 根据你的图片尺寸调整
    except ValueError:
        print(f"文件 {filename} 大小不符合预期，跳过")
        continue

    # 使用 face_recognition 检测人脸（使用CNN模型，支持GPU加速）
    face_locations = face_recognition.face_locations(Image_current, model='cnn')

    # 根据检测结果，将文件名添加到对应的列表
    if len(face_locations) > 0:
        faces_detected_files.append(filename)
    else:
        faces_not_detected_files.append(filename)

# 原始数据文件夹和目标文件夹路径
rawdata_folder = 'raw_data'
converted_data_folder = 'fillter_data'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(converted_data_folder):
    os.makedirs(converted_data_folder)

# 遍历rawdata文件夹中的所有文件
for filename in os.listdir(rawdata_folder):
    # 如果文件名在faces_detected_files列表中
    if filename in faces_detected_files:
        # 源文件路径
        source_file = os.path.join(rawdata_folder, filename)
        # 目标文件路径
        destination_file = os.path.join(converted_data_folder, filename)

        # 检查目标文件是否已经存在
        if os.path.exists(destination_file):
            print(f"文件 {filename} 已存在，跳过复制")
        else:
            # 复制文件
            shutil.copy(source_file, destination_file)
            print(f"已复制 {filename} 到 {converted_data_folder}")


faceDR_file = 'faceDR'  # 输入的faceDR文本文档路径
faceDS_file = 'faceDS'  # 输出的faceDS文本文档路径

# 打开faceDR文件进行读取，faceDS文件进行写入
with open(faceDR_file, 'r') as input_file, open(faceDS_file, 'w') as output_file:
    for line in input_file:
        # 去除行首的空格，并提取最前面的四位数字（文件名部分）
        line = line.lstrip()  # 去除最前面的空格
        index = line.split(' ')[0]  # 提取文件名部分（即前四位数字）
        if index in faces_detected_files:
            output_file.write(line)
