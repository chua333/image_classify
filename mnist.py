import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# datasets: cifar10, cifar100...
# layers: convolutional, pooling...
# models: Sequential...
from tensorflow.keras import datasets, layers, models

# read the image dataset from keras
(training_images, training_labels), (testing_images, testing_labels) = datasets.mnist.load_data()

# normalized the image to 0 and 1
training_images, testing_images = training_images / 255, testing_images / 255

# the available class in the database
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# take a look at the images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])

plt.savefig('example_mnist.png', bbox_inches='tight')
plt.show()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# training the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_images, training_labels, epochs=20, validation_data=(testing_images, testing_labels))

# plotting the graph
# plot training and validation loss values
plt.figure(figsize=(12, 6))  # Create a new figure with custom size
plt.subplot(1, 2, 1)  # Subplot for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# plot training and validation accuracy values
plt.subplot(1, 2, 2)  # Subplot for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('model_accuracy.png')
plt.show()


# model evaluation
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# saving the model
model.save('number_image_classifier.model')

# loading the model
model = models.load_model('number_image_classifier.model')

# images have to be in 32x32x3, 3 is rgb
img = cv.imread('images/0.png')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_resized = cv.resize(img_rgb, (28, 28))
img_gray = cv.cvtColor(img_resized, cv.COLOR_RGB2GRAY)
img_normalized = img_gray / 255.0
img_normalized = 1 - img_normalized
# plt.imshow(img_normalized, cmap=plt.cm.binary)
# plt.show()

img_for_prediction = img_normalized.reshape((1, 28, 28, 1))
prediction = model.predict(img_for_prediction)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
