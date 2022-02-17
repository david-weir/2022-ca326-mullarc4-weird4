# Train face mask detector model.
# Trains a neural network using the preset dataset test

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import adam_v2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import os

# initialize the initial learning rate, number of training epochs and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# accesses the test dataset folder
# initialize the list of images and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images("./test"))  # creates a list of image paths from the test dataset

# The test dataset consists of 2 folders named mask and no_mask. These names later act as labels.
# Each folder contains images used in training a model to recognise masked and unmasked faces.

data = []  # images
labels = []  # masked / unmasked

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename (masked / unmasked)
    label = imagePath.split(os.path.sep)[-2]

    # load the input image and preprocess
    image = load_img(imagePath, target_size=(224, 224))  # load image and resize to 224x224 pixels
    image = img_to_array(image)  # convert image to array
    image = preprocess_input(image)

    # update the data and labels lists
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()  # binarize labels in one-vs-all fashion
labels = lb.fit_transform(labels)  # calculate the mean + variance of features in the data and transform these features
# using this mean + var calculated
labels = to_categorical(labels)  # convert array of labeled data to one-hot vector

# partition the data into training and testing splits
# 80% of the data for training and 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
# set rotation, zoom, shear, shift and flip parameters
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# MobileNetV2 fine-tuning

# load the MobileNetV2 network with ImageNet pre-trained weights
# excluding the head FC layer (include_top:False). Excludes fully-connected layers @ the top of the network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct a new FC head of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they are
# not updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")

# compile model using Adam optimizer and binary cross-entropy
opt = adam_v2.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),  # provides batches of mutated (augmentation) image data
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image find the index of the class label with the highest probability
predIdxs = np.argmax(predIdxs, axis=1)

# Optional: show a nicely formatted classification report
# print(classification_report(testY.argmax(axis=1), predIdxs,
#                             target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")
