# 导入所需的库
from PIL import Image  # 用于图像处理
import os  # 用于文件和路径操作
import sys  # 用于系统相关操作
import numpy as np  # 用于数值计算
import time  # 用于计时
from sklearn import svm  # 导入SVM模型
from sklearn.decomposition import PCA  # 用于降维
import joblib  # 用于保存和加载模型
import warnings  # 用于警告控制

# 忽略警告信息
warnings.filterwarnings('ignore')

def get_file_list(path):
    '''
    获取指定路径下的所有jpg文件
    参数:
        path: 文件路径
    返回:
        包含所有jpg文件完整路径的列表
    '''
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"错误：路径不存在 {path}")
        return []
    # 使用列表推导式获取所有jpg文件
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    print(f"在 {path} 中找到 {len(files)} 个jpg文件")
    return files

def get_img_name_str(imgPath):
    '''
    从完整路径中获取图像文件名
    参数:
        imgPath: 图像的完整路径
    返回:
        图像文件名(不含路径)
    '''
    return imgPath.split(os.path.sep)[-1]

def img2vector(imgFile):
    '''
    将图像转换为向量
    参数:
        imgFile: 图像文件路径
    返回:
        1x784的图像向量（如果处理成功），None（如果处理失败）
    '''
    try:
        # 打开图像并转换为灰度图
        img = Image.open(imgFile).convert('L')
        # 转换为numpy数组（28x28的灰度图）
        img_arr = np.array(img, 'i')
        # 归一化处理，将像素值缩放到0-1之间
        img_normalization = np.round(img_arr / 255)
        # 将28x28的矩阵重塑为1x784的向量
        img_arr2 = np.reshape(img_normalization, (1, 784))
        return img_arr2
    except Exception as e:
        print(f"处理图片 {imgFile} 时出错: {str(e)}")
        return None

def read_and_convert(imgFileList):
    '''
    读取并转换图像文件列表为向量和标签
    参数:
        imgFileList: 图像文件路径列表
    返回:
        dataMat: 包含所有有效样本的矩阵
        dataLabel: 对应的标签列表
    '''
    if not imgFileList:
        raise ValueError("没有找到任何图像文件")

    dataLabel = []  # 存放类标签
    dataNum = len(imgFileList)  # 文件总数
    dataMat = np.zeros((dataNum, 784))  # 预分配空间，每个样本784维
    valid_samples = 0  # 有效样本计数

    # 处理每个图像文件
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        try:
            # 获取文件名和类别标签
            imgName = get_img_name_str(imgNameStr)  # 文件名格式：数字_序号.jpg
            classTag = imgName.split("_")[0]  # 提取数字作为类别标签
            vector = img2vector(imgNameStr)  # 转换为向量

            # 如果转换成功，保存数据
            if vector is not None:
                dataMat[valid_samples, :] = vector
                dataLabel.append(classTag)
                valid_samples += 1

            # 打印处理进度
            if (i + 1) % 1000 == 0:
                print(f'已处理 {i + 1}/{dataNum} 张图片')

        except Exception as e:
            print(f"处理文件 {imgNameStr} 时出错: {str(e)}")
            continue

    # 检查是否有有效样本
    if valid_samples == 0:
        raise ValueError("没有成功处理任何图像文件")

    print(f"成功处理了 {valid_samples} 张图片")
    return dataMat[:valid_samples], dataLabel

def read_all_data():
    '''
    读取所有训练数据
    返回:
        dataMat: 处理后的图像矩阵
        dataLabel: 对应的标签列表
    '''
    base_dir = r'E:\My struggle\SVM 手写数字'
    train_data_path = os.path.join(base_dir, 'img_train', 'jpg_images')

    print(f"读取训练数据从: {train_data_path}")

    # 获取所有图像文件
    flist = get_file_list(train_data_path)
    if not flist:
        raise ValueError(f"在{train_data_path}中没有找到jpg图片")

    # 转换为矩阵和标签
    dataMat, dataLabel = read_and_convert(flist)
    return dataMat, dataLabel

def reduce_dimensions(dataMat, n_components=256):
    '''
    使用PCA进行降维
    参数:
        dataMat: 原始数据矩阵
        n_components: 降维后的维度，默认256
    返回:
        降维后的数据矩阵
    '''
    print(f"正在进行PCA降维，从{dataMat.shape[1]}维降至{n_components}维...")
    start_time = time.time()

    # 创建PCA对象并执行降维
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(dataMat)

    # 计算并打印信息保留率
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"降维后保留了{explained_variance:.2f}%的信息")
    print(f"降维完成，用时：{time.time() - start_time:.2f}秒")

    # 保存PCA模型
    base_dir = r'E:\My struggle\SVM 手写数字'
    pca_path = os.path.join(base_dir, 'pca.model')
    joblib.dump(pca, pca_path)
    print(f"PCA模型已保存到：{pca_path}")

    return reduced_data

def create_svm(dataMat, dataLabel, path, decision='ovr'):
    '''
    创建并训练SVM模型
    参数:
        dataMat: 训练数据矩阵
        dataLabel: 训练标签
        path: 模型保存路径
        decision: 决策函数类型，默认'ovr'(one-vs-rest)
    返回:
        训练好的SVM分类器
    '''
    print(f"开始训练SVM模型... 数据维度: {dataMat.shape}")
    start_time = time.time()

    print("正在训练...")
    # 创建SVM分类器，使用RBF核函数
    clf = svm.SVC(decision_function_shape=decision, kernel='rbf', C=1.0)
    # 训练模型
    rf = clf.fit(dataMat, dataLabel)

    print("正在保存模型...")
    joblib.dump(rf, path)  # 保存模型

    # 打印训练信息
    training_time = time.time() - start_time
    print(f"模型训练完成，用时：{training_time:.2f}秒")
    print(f"模型已保存到：{path}")
    return clf

# 主程序入口
if __name__ == '__main__':
    try:
        print('正在运行模型请稍等')

        # 1. 读取训练数据
        dataMat, dataLabel = read_all_data()
        print(f"原始数据维度: {dataMat.shape}")

        # 2. 使用PCA进行降维
        reduced_dataMat = reduce_dimensions(dataMat, n_components=256)
        print(f"降维后数据维度: {reduced_dataMat.shape}")

        # 3. 训练并保存SVM模型
        base_dir = r'E:\My struggle\SVM 手写数字'
        model_path = os.path.join(base_dir, 'svm.model')
        create_svm(reduced_dataMat, dataLabel, model_path, decision='ovr')
        print('模型训练存储完成')

    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)  # 发生错误时退出程序