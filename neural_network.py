# neural_network.py
import numpy as np
import os
import time
from sklearn.neural_network import MLPClassifier
import joblib
import svm  # 复用之前的图像处理函数
def train_neural_network():
    '''
    训练神经网络模型
    使用与SVM相同的数据，但使用不同的模型
    '''
    print('开始训练神经网络模型...')

    try:
        # 1. 读取训练数据
        print("正在读取训练数据...")
        dataMat, dataLabel = svm.read_all_data()
        print(f"原始数据维度: {dataMat.shape}")

        # 2. PCA降维（与SVM保持一致）
        print("正在进行PCA降维...")
        start_time = time.time()

        # 创建和训练PCA模型
        from sklearn.decomposition import PCA
        pca = PCA(n_components=256)
        reduced_dataMat = pca.fit_transform(dataMat)

        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f"降维后保留了{explained_variance:.2f}%的信息")
        print(f"降维完成，用时：{time.time() - start_time:.2f}秒")

        # 3. 创建并训练神经网络
        print("开始训练神经网络...")
        start_time = time.time()

        # 创建神经网络分类器
        # hidden_layer_sizes=(100, 50) 表示两个隐藏层，分别有100和50个神经元
        clf = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',  # 使用ReLU激活函数
            solver='adam',  # 使用Adam优化器
            max_iter=50,  # 最大迭代次数
            random_state=42,  # 随机种子
            verbose=True  # 显示训练进度
        )

        # 训练模型
        clf.fit(reduced_dataMat, dataLabel)

        training_time = time.time() - start_time
        print(f"模型训练完成，用时：{training_time:.2f}秒")

        # 4. 保存模型
        base_dir = r'E:\My struggle\SVM 手写数字'

        # 保存神经网络模型
        nn_model_path = os.path.join(base_dir, 'neural_network.model')
        joblib.dump(clf, nn_model_path)
        print(f"神经网络模型已保存到：{nn_model_path}")

        # 保存PCA模型
        pca_path = os.path.join(base_dir, 'nn_pca.model')
        joblib.dump(pca, pca_path)
        print(f"PCA模型已保存到：{pca_path}")

        return clf, pca

    except Exception as e:
        print(f"训练过程中出现错误：{str(e)}")
        raise


if __name__ == '__main__':
    try:
        # 训练模型
        clf, pca = train_neural_network()
        print("模型训练和保存完成！")

    except Exception as e:
        print(f"程序执行出错：{str(e)}")