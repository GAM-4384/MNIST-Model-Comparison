import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 设置内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# 设置数据类型为 float32
tf.keras.backend.set_floatx('float32')

# 加载数据
print("Loading data...")
data = np.load('E:\\My struggle\\SVM 手写数字\\mnist.npz')
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']


# 数据预处理
def preprocess_data(x, y):
    # 转换为float32以节省内存
    x = x.astype('float32') / 255.0
    x = x.reshape(-1, 28, 28, 1)
    # One-hot编码S
    y = tf.keras.utils.to_categorical(y, 10)
    return x, y


x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

print("训练集形状:", x_train.shape, y_train.shape)
print("测试集形状:", x_test.shape, y_test.shape)

# 创建数据集
BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(10000) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)


# 构建模型
def create_model():
    model = tf.keras.Sequential([
        # 第一个卷积块
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # 第二个卷积块
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # 展平层和全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


# 创建模型
model = create_model()

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型结构
model.summary()

# 定义回调函数
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# 训练模型
print("\nTraining model...")
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# 评估模型
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 保存模型
model.save('mnist_cnn_model.h5')
print("Model saved as 'mnist_cnn_model.h5'")


# 可视化训练过程
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(history)


# 预测函数
def predict_samples(model, test_dataset, num_samples=5):
    # 获取一批测试数据
    for images, labels in test_dataset.take(1):
        images = images[:num_samples]
        labels = labels[:num_samples]

        # 预测
        predictions = model.predict(images)

        # 显示结果
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
            pred_label = np.argmax(predictions[i])
            true_label = np.argmax(labels[i])
            confidence = predictions[i][pred_label]

            # 设置标题颜色
            color = 'green' if pred_label == true_label else 'red'
            plt.title(f'Pred: {pred_label}\nTrue: {true_label}\n{confidence * 100:.1f}%',
                      color=color)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 打印详细信息
        print("\n预测详细信息：")
        print("样本  预测值  真实值  置信度    结果")
        print("-" * 40)
        for i in range(num_samples):
            pred_label = np.argmax(predictions[i])
            true_label = np.argmax(labels[i])
            confidence = predictions[i][pred_label]
            result = "✓" if pred_label == true_label else "✗"
            print(f"{i + 1:2d}    {pred_label:3d}    {true_label:3d}    {confidence * 100:6.2f}%  {result}")


# 显示一些预测结果
print("\nDisplaying predictions...")
predict_samples(model, test_dataset)