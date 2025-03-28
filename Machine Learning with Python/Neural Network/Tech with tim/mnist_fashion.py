import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Dataset Loading
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Decrease size
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display dataset
# print(train_images[7])
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels)

print('Test Acc:',test_acc)

# Predictions
predictions = model.predict(test_images)
# print(class_names[np.argmax(predictions[0])])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual:"+ class_names[test_labels[i]])
    plt.title("Prediction:"+ class_names[np.argmax(predictions[i])])
    plt.show()
    
predictions1 = model.predict([test_images[7]])

