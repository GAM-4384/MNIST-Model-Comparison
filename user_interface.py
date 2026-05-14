# 步骤一: 导入模块
import sys
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import os
import joblib
# 调用自己创建的类
import svm
import numpy as np


# 步骤二: 创建类，完成可视化窗口的初始化
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        # 设置窗口大小
        Dialog.resize(645, 475)

        # 设置"打开图像"按钮
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(230, 340, 141, 41))
        self.pushButton.setAutoDefault(False)
        self.pushButton.setObjectName("pushButton")

        # 设置"显示标签"按钮
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(220, 50, 191, 221))
        self.label.setWordWrap(False)
        self.label.setObjectName("label")

        # 设置文本编辑区域
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(220, 280, 191, 41))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    # 创建窗口设置
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "手写体识别"))
        self.pushButton.setText(_translate("Dialog", "打开图像"))
        self.label.setText(_translate("Dialog", "显示图像"))


# 步骤三: 创建类，完成测试集图像验证功能
class MyWindow(QMainWindow, Ui_Dialog):
    # 初始化数据
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 加载模型
        self.load_models()
        self.pushButton.clicked.connect(self.openImage)  # 点击事件，开启下面的函数

    def load_models(self):
        """加载SVM和PCA模型"""
        base_dir = r'E:\My struggle\SVM 手写数字'
        # 加载SVM模型
        model_path = os.path.join(base_dir, 'svm.model')
        self.clf = joblib.load(model_path)
        # 加载PCA模型
        pca_path = os.path.join(base_dir, 'pca.model')
        self.pca = joblib.load(pca_path)
        print("模型加载完成")

    # 点击事件函数
    def openImage(self):
        try:
            # 点击"打开图像"按钮时
            imgName, imgType = QFileDialog.getOpenFileName(self, "打开图像",
                                                           r"E:\My struggle\SVM 手写数字\img_test\jpg_images",
                                                           "图像文件 (*.jpg)")
            if not imgName:
                return

            # 获取图像宽高，显示在对话框上
            png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(png)

            # 处理图像并预测
            print(f"正在处理图像: {imgName}")
            # 转换图像为向量
            dataMat = svm.img2vector(imgName)
            if dataMat is None:
                self.show_result("图像处理失败")
                return

            # 使用PCA降维
            reduced_data = self.pca.transform(dataMat)
            # 预测结果
            preResult = self.clf.predict(reduced_data)

            # 显示结果
            self.show_result(preResult[0])

        except Exception as e:
            print(f"错误：{str(e)}")
            self.show_result("处理过程出错")

    def show_result(self, result):
        """显示预测结果"""
        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet("color:red")
        self.textEdit.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.textEdit.setFontPointSize(9)
        self.textEdit.setText("预测的结果是:")
        self.textEdit.append(str(result))


# 步骤四: 主函数处理
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        myWin = MyWindow()
        myWin.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序出错：{str(e)}")