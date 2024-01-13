# 导入需要的库
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, datasets
import numpy as np

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 将图像数据转换为浮点数，并归一化到[0,1]范围
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 为了适应Lenet-5的输入要求，将图像数据从28x28扩展为32x32，并增加一个通道维度
x_train = tf.pad(x_train, [[0,0],[2,2],[2,2]])
x_test = tf.pad(x_test, [[0,0],[2,2],[2,2]])
x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

# 将标签数据转换为one-hot编码
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# 定义一个函数，用于构建不同激活函数和dropout率的Lenet-5模型
def build_model(activation='relu', dropout_rate=0.5):
  model = models.Sequential([
    # 第一个卷积层，使用6个5x5的卷积核，激活函数由参数指定
    layers.Conv2D(6, kernel_size=5, activation=activation, input_shape=(32, 32, 1)),
    # 第一个池化层，使用2x2的最大池化
    layers.MaxPool2D(pool_size=2),
    # 第二个卷积层，使用16个5x5的卷积核，激活函数由参数指定
    layers.Conv2D(16, kernel_size=5, activation=activation),
    # 第二个池化层，使用2x2的最大池化
    layers.MaxPool2D(pool_size=2),
    # 将卷积层的输出展平为一维向量
    layers.Flatten(),
    # 第一个全连接层，有120个神经元，激活函数由参数指定
    layers.Dense(120, activation=activation),
    # 在全连接层后添加dropout层，丢弃率由参数指定
    layers.Dropout(dropout_rate),
    # 第二个全连接层，有84个神经元，激活函数由参数指定
    layers.Dense(84, activation=activation),
    # 在全连接层后添加dropout层，丢弃率由参数指定
    layers.Dropout(dropout_rate),
    # 第三个全连接层，有10个神经元，激活函数为softmax，输出分类概率
    layers.Dense(10, activation='softmax')
  ])
  return model

# 定义一个函数，用于训练和评估不同数据量和超参数的模型
def train_and_evaluate_model(model, batch_size=128, learning_rate=0.01, epochs=10):
  # 编译模型，使用交叉熵损失函数和Adam优化器，评估指标为准确率
  model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
  # 训练模型，使用指定的批量大小和周期数，每个周期结束后在测试集上评估模型性能，并记录训练过程中的损失和准确率
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
  # 打印模型在测试集上的准确率
  test_loss, test_acc = model.evaluate(x_test, y_test)
  print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_acc))
  # 使用模型对测试集进行预测，并计算混淆矩阵
  y_pred = model.predict(x_test)
  y_pred = np.argmax(y_pred, axis=1)
  y_true = np.argmax(y_test, axis=1)
  confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
  print('Confusion matrix:\n', confusion_matrix.numpy())

# # 使用sigmoid激活函数和0.5的dropout率构建模型
# model_sigmoid = build_model(activation='sigmoid', dropout_rate=0.5)
# # 使用128的批量大小，0.01的学习率和10个周期训练和评估模型
# train_and_evaluate_model(model_sigmoid, batch_size=128, learning_rate=0.01, epochs=1)

# # 使用tanh激活函数和0.5的dropout率构建模型
# model_tanh = build_model(activation='tanh', dropout_rate=0.5)
# # 使用128的批量大小，0.01的学习率和10个周期训练和评估模型
# train_and_evaluate_model(model_tanh, batch_size=128, learning_rate=0.01, epochs=1)

# # 使用ReLU激活函数和0.5的dropout率构建模型
# model_relu = build_model(activation='relu', dropout_rate=0.5)
# # 使用128的批量大小，0.01的学习率和10个周期训练和评估模型
# train_and_evaluate_model(model_relu, batch_size=128, learning_rate=0.01, epochs=1)

# # 定义一个列表，存储不同的dropout率
# dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
# # 定义一个列表，存储不同dropout率对应的测试准确率
# test_accs = []
# # 对每个dropout率，构建、训练和评估模型
# for dropout_rate in dropout_rates:
#   # 使用ReLU激活函数和指定的dropout率构建模型
#   model = build_model(activation='relu', dropout_rate=dropout_rate)
#   # 使用128的批量大小，0.01的学习率和3个周期训练和评估模型
#   train_and_evaluate_model(model, batch_size=128, learning_rate=0.01, epochs=1)
#   # 记录测试准确率
#   test_acc = model.evaluate(x_test, y_test)[1]
#   test_accs.append(test_acc)
# 定义一个列表，存储不同的数据量
data_sizes = [10000, 20000, 30000, 40000, 50000, 60000]
# 定义一个列表，存储不同数据量对应的测试准确率
test_accs = []
# 对每个数据量，构建、训练和评估模型
for data_size in data_sizes:
  # 使用ReLU激活函数和0.5的dropout率构建模型
  model = build_model(activation='relu', dropout_rate=0.5)
  # 使用128的批量大小，0.01的学习率和10个周期训练和评估模型，只使用指定数量的训练数据
  train_and_evaluate_model(model, batch_size=128, learning_rate=0.01, epochs=10, x_train=x_train[:data_size], y_train=y_train[:data_size])
  # 记录测试准确率
  test_acc = model.evaluate(x_test, y_test)[1]
  test_accs.append(test_acc)


