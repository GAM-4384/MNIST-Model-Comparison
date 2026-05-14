# test_neural_network.py
import sys
import time
import os
import joblib
import numpy as np
import svm  # 复用之前的图像处理函数
def load_neural_network_models():
    '''
    加载训练好的神经网络和PCA模型
    返回:
        clf: 训练好的神经网络分类器
        pca: 训练好的PCA模型
    '''
    base_dir = r'E:\My struggle\SVM 手写数字'

    # 加载神经网络模型
    nn_model_path = os.path.join(base_dir, 'neural_network.model')
    print(f"正在加载神经网络模型：{nn_model_path}")
    clf = joblib.load(nn_model_path)

    # 加载PCA模型
    pca_path = os.path.join(base_dir, 'nn_pca.model')
    print(f"正在加载PCA模型：{pca_path}")
    pca = joblib.load(pca_path)

    return clf, pca


def prepare_test_data():
    '''
    准备测试数据
    返回:
        tdataMat: 测试数据矩阵
        tdataLabel: 测试数据标签
    '''
    base_dir = r'E:\My struggle\SVM 手写数字'
    test_path = os.path.join(base_dir, 'img_test', 'jpg_images')
    print(f"读取测试数据从：{test_path}")

    # 读取所有图像
    tflist = svm.get_file_list(test_path)
    if not tflist:
        raise ValueError(f"在{test_path}中没有找到jpg图片")

    # 转换为矩阵和标签
    tdataMat, tdataLabel = svm.read_and_convert(tflist)
    print("原始测试集数据维度为:{0}，标签数量:{1}".format(tdataMat.shape, len(tdataLabel)))

    return tdataMat, tdataLabel


def evaluate_neural_network(clf, pca, test_data, test_labels):
    '''
    评估神经网络模型性能
    参数:
        clf: 神经网络分类器
        pca: PCA模型
        test_data: 测试数据
        test_labels: 测试标签
    返回:
        准确率
    '''
    print("\n开始评估神经网络模型...")

    # PCA降维
    print("对测试数据进行PCA降维...")
    reduced_test_data = pca.transform(test_data)
    print(f"降维后测试集维度: {reduced_test_data.shape}")

    # 计算准确率
    score_start = time.time()
    score = clf.score(reduced_test_data, test_labels)
    score_end = time.time()

    print("计算准确率花费 {:.2f} 秒".format(score_end - score_start))
    print("准确率: {:.2%}".format(score))
    print("错误率: {:.2%}".format(1 - score))

    return score


if __name__ == '__main__':
    try:
        print("开始测试神经网络模型...")
        test_start = time.time()

        # 加载模型
        clf, pca = load_neural_network_models()

        # 准备测试数据
        test_data, test_labels = prepare_test_data()

        # 评估模型
        accuracy = evaluate_neural_network(clf, pca, test_data, test_labels)

        test_end = time.time()
        print("\n测试总耗时 {:.2f} 秒".format(test_end - test_start))

    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)