from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)


def show_number(X_train_or_test, y_train_or_test, x):
    print("Label:", y_train_or_test[x])
    plt.imshow(X_train_or_test[x], cmap='Greys')
show_number(X_train, y_train, 0)

X_train_images = X_train.reshape(60000, 784)
X_test_images = X_test.reshape(10000, 784)
X_train_images = X_train_images.astype('float32')
X_test_images = X_test_images.astype('float32')
X_train_images /= 255
X_test_images /= 255

y_train_labels = keras.utils.to_categorical(y_train, 10)
y_test_labels = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (784,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train_images, y_train_labels, batch_size = 100, epochs = 10, verbose = 2, 
                    validation_data = (X_test_images, y_test_labels))

score = model.evaluate(X_test_images, y_test_labels, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# prints wrongly recognized numbers
for i in range(1000):
    X_test_images1 = X_test_images[i, :].reshape(1, 784)
    predicted_category = model.predict(X_test_images1).argmax()
    label = y_test_labels[i].argmax()
    if (predicted_category != label):
        plt.title("Predicted: %d Label: %d" % (predicted_category, label))
        plt.imshow(X_test_images1.reshape([28, 28]), cmap = "Greys")
        plt.show()


