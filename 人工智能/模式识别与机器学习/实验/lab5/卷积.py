import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

# 加载数据集
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 对属性进行归一化，使它的取值在0-1之间，同时转换为tensor张量，类型为tf.flost32
X_train = train_x.reshape(60000, 28, 28, 1)
X_test = test_x.reshape(10000, 28, 28, 1)

X_train, X_test = tf.cast(X_train / 255.0, tf.float32), tf.cast(X_test / 255.0, tf.float32)
y_train, y_test = tf.cast(train_y, tf.int32), tf.cast(test_y, tf.int32)

# 建立模型
model = tf.keras.Sequential([
    # unit1
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    # unit2
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    # unit3
    tf.keras.layers.Flatten(),

    # unit4
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 配置训练方法
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# 训练模型
history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# 评估模型
model.evaluate(X_test, y_test, verbose=2)
pd.DataFrame(history.history).to_csv("training_log.csv", index=False)
graph = pd.read_csv('training_log.csv')

# 使用模型
for i in range(10):
    num = np.random.randint(1, 10000)

    plt.subplot(2, 5, i + 1)
    plt.axis("off")
    plt.imshow(test_x[num], cmap="gray")
    demo = tf.reshape(X_test[num], (1, 28, 28, 1))
    y_pred = np.argmax(model.predict(demo))
    plt.title("y=" + str(test_y[num]) + "\ny_pred" + str(y_pred))

plt.show()
