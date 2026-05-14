# 导入必要的库
import numpy as np  # 用于数值计算和数组操作
import struct  # 用于解析二进制数据
import os  # 用于文件和目录操作
from PIL import Image  # 用于图像处理


def decode_idx3_ubyte(filename):
    '''
    解码idx3-ubyte文件，即MNIST图像文件
    参数:
        filename: idx3-ubyte文件路径
    返回:
        图像数据的numpy数组
    '''
    # 读取二进制数据
    bin_data = open(filename, 'rb').read()

    # 解析文件头信息
    # '>iiii'表示使用大端存储，读取4个整数
    # 分别是魔数、图片数量、图片高度、图片宽度
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数：{}，图片数量：{}，图片大小：{}*{}'.format(magic_number, num_images, num_rows, num_cols))

    # 解析图像数据
    image_size = num_rows * num_cols  # 计算每张图片的像素总数
    offset += struct.calcsize(fmt_header)  # 移过文件头
    fmt_image = '>' + str(image_size) + 'B'  # 'B'表示无符号字符
    images = np.empty((num_images, num_rows, num_cols))  # 预分配空间

    # 逐张解析图片
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, bin_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images


def decode_idx1_ubyte(filename):
    '''
    解码idx1-ubyte文件，即MNIST标签文件
    参数:
        filename: idx1-ubyte文件路径
    返回:
        标签数据的numpy数组
    '''
    # 读取二进制数据
    bin_data = open(filename, 'rb').read()

    # 解析文件头信息，包含魔数和标签数量
    offset = 0
    fmt_header = '>ii'  # 两个整数
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数：{}，标签数量：{}'.format(magic_number, num_images))

    # 解析标签数据
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'  # 一个字节的无符号整数
    labels = np.empty(num_images)  # 预分配空间

    # 逐个解析标签
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def save_images(images, labels, save_dir):
    '''
    将图像数据保存为jpg文件
    参数:
        images: 图像数据数组
        labels: 标签数组
        save_dir: 保存目录
    '''
    # 如果保存目录不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 逐张保存图片
    for i in range(len(images)):
        # 创建图像对象，uint8表示8位无符号整数
        img = Image.fromarray(images[i].astype('uint8'))
        # 构造文件名：标签_序号.jpg
        file_name = os.path.join(save_dir, '{}_{}.jpg'.format(int(labels[i]), i))
        # 保存图片
        img.save(file_name)

        # 每保存1000张打印进度
        if (i + 1) % 1000 == 0:
            print('已保存 {} 张图片'.format(i + 1))


def convert_mnist_to_jpg(images_file, labels_file, save_dir, dataset_type="训练集"):
    '''
    转换MNIST数据为jpg图片的主函数
    参数:
        images_file: 图像数据文件路径
        labels_file: 标签数据文件路径
        save_dir: 保存目录
        dataset_type: 数据集类型（训练集/测试集）
    '''
    print(f'开始转换MNIST{dataset_type}数据...')

    # 读取数据
    print(f'读取{dataset_type}图片...')
    images = decode_idx3_ubyte(images_file)
    print(f'读取{dataset_type}标签...')
    labels = decode_idx1_ubyte(labels_file)

    # 保存图片
    print(f'保存{dataset_type}图片到 {save_dir}...')
    save_images(images, labels, save_dir)

    print(f'{dataset_type}转换完成！')


if __name__ == '__main__':
    # 设置基础路径
    base_dir = r'E:\My struggle\SVM 手写数字'

    # 设置训练集相关路径
    train_images_file = os.path.join(base_dir, 'img_train', 'train-images-idx3-ubyte', 'train-images.idx3-ubyte')
    train_labels_file = os.path.join(base_dir, 'img_train', 'train-labels-idx1-ubyte', 'train-labels.idx1-ubyte')
    train_save_dir = os.path.join(base_dir, 'img_train', 'jpg_images')

    # 设置测试集相关路径
    test_images_file = os.path.join(base_dir, 'img_test', 't10k-images-idx3-ubyte', 't10k-images.idx3-ubyte')
    test_labels_file = os.path.join(base_dir, 'img_test', 't10k-labels-idx1-ubyte', 't10k-labels.idx1-ubyte')
    test_save_dir = os.path.join(base_dir, 'img_test', 'jpg_images')

    # 打印所有路径信息，方便调试
    print('训练集图像文件:', train_images_file)
    print('训练集标签文件:', train_labels_file)
    print('训练集保存目录:', train_save_dir)
    print('测试集图像文件:', test_images_file)
    print('测试集标签文件:', test_labels_file)
    print('测试集保存目录:', test_save_dir)

    try:
        # 检查并处理训练集文件
        if os.path.exists(train_images_file) and os.path.exists(train_labels_file):
            convert_mnist_to_jpg(train_images_file, train_labels_file, train_save_dir, "训练集")
        else:
            print('警告：训练集文件不存在，跳过训练集处理')

        # 检查并处理测试集文件
        if os.path.exists(test_images_file) and os.path.exists(test_labels_file):
            convert_mnist_to_jpg(test_images_file, test_labels_file, test_save_dir, "测试集")
        else:
            print('警告：测试集文件不存在，跳过测试集处理')

    except Exception as e:
        print(f'转换过程中出现错误：{str(e)}')