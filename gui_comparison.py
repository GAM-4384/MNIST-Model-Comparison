# gui_comparison.py
import sys
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox, QProgressDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import joblib
import svm
import numpy as np
import time

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        # 创建中央窗口部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # 创建主布局
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralwidget)

        # 创建图像显示区域
        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setMinimumSize(200, 200)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setText("显示图像")
        self.mainLayout.addWidget(self.imageLabel)

        # 创建结果显示区域
        self.resultsGroup = QtWidgets.QGroupBox("预测结果", self.centralwidget)
        self.resultsLayout = QtWidgets.QHBoxLayout()

        # SVM结果
        self.svmGroup = QtWidgets.QGroupBox("SVM模型")
        self.svmLayout = QtWidgets.QVBoxLayout()
        self.svmResult = QtWidgets.QTextEdit()
        self.svmResult.setReadOnly(True)
        self.svmResult.setMaximumHeight(100)
        self.svmLayout.addWidget(self.svmResult)
        self.svmGroup.setLayout(self.svmLayout)

        # 神经网络结果
        self.nnGroup = QtWidgets.QGroupBox("神经网络模型")
        self.nnLayout = QtWidgets.QVBoxLayout()
        self.nnResult = QtWidgets.QTextEdit()
        self.nnResult.setReadOnly(True)
        self.nnResult.setMaximumHeight(100)
        self.nnLayout.addWidget(self.nnResult)
        self.nnGroup.setLayout(self.nnLayout)

        # 添加到结果布局
        self.resultsLayout.addWidget(self.svmGroup)
        self.resultsLayout.addWidget(self.nnGroup)
        self.resultsGroup.setLayout(self.resultsLayout)
        self.mainLayout.addWidget(self.resultsGroup)

        # 创建性能统计区域
        self.statsGroup = QtWidgets.QGroupBox("模型性能统计", self.centralwidget)
        self.statsLayout = QtWidgets.QVBoxLayout()
        self.statsText = QtWidgets.QTextEdit()
        self.statsText.setReadOnly(True)
        self.statsText.setMaximumHeight(100)
        self.statsLayout.addWidget(self.statsText)
        self.statsGroup.setLayout(self.statsLayout)
        self.mainLayout.addWidget(self.statsGroup)

        # 创建按钮
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.openButton = QtWidgets.QPushButton("打开图像", self.centralwidget)
        self.testAllButton = QtWidgets.QPushButton("测试所有样本", self.centralwidget)
        self.buttonLayout.addWidget(self.openButton)
        self.buttonLayout.addWidget(self.testAllButton)
        self.mainLayout.addLayout(self.buttonLayout)

        # 设置窗口标题
        MainWindow.setWindowTitle("手写体识别模型对比")

        # 设置菜单栏和状态栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

class ComparisonWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ComparisonWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.load_models()

        # 连接按钮信号
        self.ui.openButton.clicked.connect(self.openImage)
        self.ui.testAllButton.clicked.connect(self.testAllSamples)

        # 初始化性能统计
        self.svm_correct = 0
        self.nn_correct = 0
        self.total_samples = 0

    def load_models(self):
        '''加载所有模型'''
        try:
            base_dir = r'E:\My struggle\SVM 手写数字'

            # 加载SVM模型
            model_path = os.path.join(base_dir, 'svm.model')
            self.svm_clf = joblib.load(model_path)
            self.svm_pca = joblib.load(os.path.join(base_dir, 'pca.model'))

            # 加载神经网络模型
            nn_model_path = os.path.join(base_dir, 'neural_network.model')
            self.nn_clf = joblib.load(nn_model_path)
            self.nn_pca = joblib.load(os.path.join(base_dir, 'nn_pca.model'))

            print("所有模型加载完成")

        except Exception as e:
            print(f"模型加载错误：{str(e)}")
            QMessageBox.critical(self, "错误", "模型加载失败！")

    def openImage(self):
        '''打开并处理单个图像'''
        try:
            # 打开文件对话框
            imgName, _ = QFileDialog.getOpenFileName(
                self,
                "打开图像",
                r"E:\My struggle\SVM 手写数字\img_test\jpg_images",
                "图像文件 (*.jpg)"
            )

            if not imgName:
                return

            # 显示图像
            pixmap = QtGui.QPixmap(imgName).scaled(
                self.ui.imageLabel.width(),
                self.ui.imageLabel.height(),
                QtCore.Qt.KeepAspectRatio
            )
            self.ui.imageLabel.setPixmap(pixmap)

            # 处理图像
            dataMat = svm.img2vector(imgName)
            if dataMat is None:
                raise ValueError("图像处理失败")

            # 获取真实标签
            true_label = imgName.split('_')[0].split(os.path.sep)[-1]

            # SVM预测
            svm_data = self.svm_pca.transform(dataMat)
            svm_result = self.svm_clf.predict(svm_data)[0]

            # 神经网络预测
            nn_data = self.nn_pca.transform(dataMat)
            nn_result = self.nn_clf.predict(nn_data)[0]

            # 显示结果
            self.showResults(true_label, svm_result, nn_result)

        except Exception as e:
            print(f"图像处理错误：{str(e)}")
            QMessageBox.warning(self, "警告", "图像处理失败！")

    # 修改 ComparisonWindow 类中的 testAllSamples 方法

    def testAllSamples(self):
        '''测试所有测试集样本'''
        try:
            # 重置统计
            self.svm_correct = 0
            self.nn_correct = 0
            self.total_samples = 0

            # 准备测试数据
            test_path = r'E:\My struggle\SVM 手写数字\img_test\jpg_images'

            # 获取所有jpg文件
            test_files = []
            for file in os.listdir(test_path):
                if file.endswith('.jpg'):
                    test_files.append(os.path.join(test_path, file))

            if not test_files:
                QMessageBox.warning(self, "警告", f"在路径 {test_path} 中未找到任何jpg文件！")
                return

            # 创建进度对话框
            progress = QProgressDialog("正在测试...", "取消", 0, len(test_files), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)

            # 处理所有图像
            for i, imgFile in enumerate(test_files):
                try:
                    progress.setValue(i)
                    if progress.wasCanceled():
                        break

                    # 获取真实标签 (从文件名中提取)
                    filename = os.path.basename(imgFile)
                    true_label = filename.split('_')[0]

                    # 处理图像
                    dataMat = svm.img2vector(imgFile)
                    if dataMat is None:
                        print(f"无法处理图像: {imgFile}")
                        continue

                    # SVM预测
                    svm_data = self.svm_pca.transform(dataMat)
                    svm_result = str(self.svm_clf.predict(svm_data)[0])

                    # 神经网络预测
                    nn_data = self.nn_pca.transform(dataMat)
                    nn_result = str(self.nn_clf.predict(nn_data)[0])

                    # 更新统计
                    self.total_samples += 1
                    if svm_result == true_label:
                        self.svm_correct += 1
                    if nn_result == true_label:
                        self.nn_correct += 1

                    # 定期打印进度
                    if i % 100 == 0:
                        print(f"已处理: {i}/{len(test_files)}, "
                              f"SVM正确率: {self.svm_correct / self.total_samples * 100:.2f}%, "
                              f"NN正确率: {self.nn_correct / self.total_samples * 100:.2f}%")

                except Exception as e:
                    print(f"处理文件 {imgFile} 时出错: {str(e)}")
                    continue

            progress.setValue(len(test_files))

            # 显示最终统计结果
            self.updateStats()

            # 显示详细的结果对话框
            QMessageBox.information(self, "测试完成",
                                    f"测试完成！\n"
                                    f"总样本数: {self.total_samples}\n"
                                    f"SVM正确数: {self.svm_correct} (准确率: {self.svm_correct / self.total_samples * 100:.2f}%)\n"
                                    f"神经网络正确数: {self.nn_correct} (准确率: {self.nn_correct / self.total_samples * 100:.2f}%)")

        except Exception as e:
            print(f"批量测试错误：{str(e)}")
            QMessageBox.warning(self, "警告", f"批量测试失败！错误信息：{str(e)}")

    def showResults(self, true_label, svm_result, nn_result):
        '''显示单个样本的预测结果'''
        # 设置SVM结果
        self.ui.svmResult.setStyleSheet("color: black")
        self.ui.svmResult.setText(f"真实标签: {true_label}\n预测结果: {svm_result}")
        if svm_result == true_label:
            self.ui.svmResult.append("预测正确 ✓")
            self.ui.svmResult.setStyleSheet("color: green")
        else:
            self.ui.svmResult.append("预测错误 ✗")
            self.ui.svmResult.setStyleSheet("color: red")

        # 设置神经网络结果
        self.ui.nnResult.setStyleSheet("color: black")
        self.ui.nnResult.setText(f"真实标签: {true_label}\n预测结果: {nn_result}")
        if nn_result == true_label:
            self.ui.nnResult.append("预测正确 ✓")
            self.ui.nnResult.setStyleSheet("color: green")
        else:
            self.ui.nnResult.append("预测错误 ✗")
            self.ui.nnResult.setStyleSheet("color: red")

    def updateStats(self):
        '''更新性能统计信息'''
        if self.total_samples == 0:
            return

        svm_accuracy = self.svm_correct / self.total_samples * 100
        nn_accuracy = self.nn_correct / self.total_samples * 100

        stats_text = f"测试样本总数: {self.total_samples}\n"
        stats_text += f"SVM正确数: {self.svm_correct}, 准确率: {svm_accuracy:.2f}%\n"
        stats_text += f"神经网络正确数: {self.nn_correct}, 准确率: {nn_accuracy:.2f}%"

        self.ui.statsText.setText(stats_text)

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        myWin = ComparisonWindow()
        myWin.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序出错：{str(e)}")