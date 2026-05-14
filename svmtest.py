# 导入必要的库
import sys  # 用于系统相关操作
import time  # 用于计时
import svm  # 导入自己创建的SVM模块
import os  # 用于文件和路径操作
import joblib  # 用于加载模型
import numpy as np  # 用于数值计算

def load_models():
    '''
    加载训练好的SVM模型和PCA模型
    返回:
        clf: 训练好的SVM分类器
        pca: 训练好的PCA模型
    '''
    # 设置基础路径
    base_dir = r'E:\My struggle\SVM 手写数字'

    # 加载SVM模型
    model_path = os.path.join(base_dir, 'svm.model')
    print(f"正在加载SVM模型：{model_path}")
    clf = joblib.load(model_path)

    # 加载PCA模型
    pca_path = os.path.join(base_dir, 'pca.model')
    print(f"正在加载PCA模型：{pca_path}")
    pca = joblib.load(pca_path)

    return clf, pca


def prepare_test_data():
    '''
    准备测试数据，包括读取测试图像和转换为向量
    返回:
        tdataMat: 测试数据矩阵
        tdataLabel: 测试数据标签
    '''
    # 设置测试数据路径
    base_dir = r'E:\My struggle\SVM 手写数字'
    test_path = os.path.join(base_dir, 'img_test', 'jpg_images')
    print(f"读取测试数据从：{test_path}")

    # 获取所有测试图像文件
    tflist = svm.get_file_list(test_path)
    if not tflist:
        raise ValueError(f"在{test_path}中没有找到jpg图片")

    # 将图像转换为矩阵和标签
    tdataMat, tdataLabel = svm.read_and_convert(tflist)
    print("原始测试集数据维度为:{0}，标签数量:{1}".format(tdataMat.shape, len(tdataLabel)))

    return tdataMat, tdataLabel


def evaluate_model(clf, pca, test_data, test_labels):
    '''
    评估模型性能
    参数:
        clf: SVM分类器
        pca: PCA模型
        test_data: 测试数据矩阵
        test_labels: 测试数据标签
    返回:
        score: 模型准确率
    '''
    print("\n开始评估模型...")

    # 使用PCA对测试数据进行降维
    print("对测试数据进行PCA降维...")
    reduced_test_data = pca.transform(test_data)
    print(f"降维后测试集维度: {reduced_test_data.shape}")

    # 计算模型在测试集上的准确率
    score_start = time.time()
    score = clf.score(reduced_test_data, test_labels)
    score_end = time.time()

    # 打印评估结果
    print("计算准确率花费 {:.2f} 秒".format(score_end - score_start))
    print("准确率: {:.2%}".format(score))  # 使用百分比格式显示准确率
    print("错误率: {:.2%}".format(1 - score))

    return score


# 主程序入口
if __name__ == '__main__':
    try:
        print("开始测试...")
        test_start = time.time()  # 记录测试开始时间

        # 1. 加载训练好的模型
        clf, pca = load_models()

        # 2. 准备测试数据
        test_data, test_labels = prepare_test_data()

        # 3. 评估模型性能
        accuracy = evaluate_model(clf, pca, test_data, test_labels)

        # 4. 打印总耗时
        test_end = time.time()
        print("\n测试总耗时 {:.2f} 秒".format(test_end - test_start))

    except Exception as e:
        # 异常处理
        print(f"错误：{str(e)}")
        sys.exit(1)  # 发生错误时退出程序