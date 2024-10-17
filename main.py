import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Нормализация данных (приводим значения пикселей к диапазону [0,1])
X_train, X__test = X_train / 255.0, X_test / 255.0

# Визуализация примера данных
plt.imshow(X_train[1], cmap="gray")
plt.title(f"Label: {Y_train[1]}")
plt.show()

# Построение модели
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(
            input_shape=(28, 28)
        ),  # Преобразование картинки 28 на 28 в плоский вектор
        tf.keras.layers.Dense(
            128, activation="relu"
        ),  # Полносвязный слой с 128 нейронами и ReLU активацией
        tf.keras.layers.Dense(
            10, activation="softmax"
        ),  # Выходной слой с 10 классами (от 0 до 9)
    ]
)

# Компиляция модели
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# Обучение модели
model.fit(X_train, Y_train, epochs=5, verbose=2)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Точность модели на тестовых данных: {test_acc}")

# Предсказание значения нулевого элемента  тестового датасета
predicted = model.predict(X_test[0].reshape(1, 28, 28))
print(f"Предсказанное значение для элемента: {predicted}")
# Визуализация элемента
plt.imshow(X_test[0], cmap="gray")
plt.title(f"Label: {Y_test[0]}")
plt.show()
