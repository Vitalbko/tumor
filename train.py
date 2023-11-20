import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

# libraries required to build the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from keras.activations import relu, softmax

# optimizer library
from keras.optimizers import Adam

from numpy.random import seed

seed(1)

tf.random.set_seed(2)

train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    rotation_range=2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True)

training_set = train_data_gen.flow_from_directory(directory="./dataset/Training",
                                                  target_size=(224, 224),
                                                  class_mode='categorical',
                                                  batch_size=32)

validation_data_gen = ImageDataGenerator(rescale=1. / 255)
# importing our validation set images with the same image size and batch size
validation_set = validation_data_gen.flow_from_directory(directory="./dataset/Testing",
                                                         target_size=(224, 224),
                                                         class_mode='categorical',
                                                         batch_size=32)
# next function iterates over the training set and separates the images and their labels
imgs, labels = next(training_set)


# we define a function that prints the first 10 images from the training set
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# calling our function to print the first 10 images
plotImages(imgs)
# printing all the labels from the first batch of 32 images
print(labels)

# A sequential model is a model which consists of a sequence of layers.
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=3,
                 activation=relu,
                 input_shape=[224, 224, 3]))
# Adding a secong convolutional layer
model.add(Conv2D(filters=32,
                 kernel_size=3,
                 activation=relu))
model.add(MaxPool2D(pool_size=2,
                    strides=2,
                    padding='valid'))
# adding another Conv2D layer
model.add(Conv2D(filters=32,
                 kernel_size=3,
                 activation=relu))
# adding a fourth Conv2D layer
model.add(Conv2D(filters=64,
                 kernel_size=3,
                 activation=relu))
# adding a second MaxPool2D layer
model.add(MaxPool2D(pool_size=2,
                    strides=2,
                    padding='valid'))
model.add(Flatten())
model.add(Dense(units=32,
                activation=relu,
                use_bias=True
                ))
model.add(Dropout(0.4))
# adding another fully connected layer
model.add(Dense(units=16,
                activation=relu,
                use_bias=True
                ))
model.add(Dense(units=4,
                activation=softmax))
model.summary()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model_history = model.fit(x=training_set, validation_data=validation_set, epochs=60, verbose=1)

print(model_history.history.keys())

# comparing the training and testing accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Accuracy of the model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# comparing training and testing loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Loss of the model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

index = ['glioma', 'meningioma', 'normal', 'adenoma']

test_image1 = load_img('./dataset/Testing/meningioma/Te-me_0018.jpg', target_size=(224, 224))
test_image1 = img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis=0)
result1 = np.argmax(model.predict(test_image1 / 255.0), axis=1)
print(index[result1[0]])

test_image2 = load_img('./dataset/Testing/pituitary/Te-pi_0018.jpg', target_size=(224, 224))
test_image2 = img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
result2 = np.argmax(model.predict(test_image2 / 255.0), axis=1)
print(index[result2[0]])

model.save('weights.h5')
