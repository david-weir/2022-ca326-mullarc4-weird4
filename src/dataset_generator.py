# Create an image database for testing
# Capture facial images from a webcam and store them into datasets

import cv2, os, sys

# each persons dataset is stored in this parent directory
datasets = 'datasets'

# creates a sub-folder of stored images for a given named person
# name of sub-folder / folder name corresponds with the name of the person who's images are stored in the subfolder

try:
    subdata = sys.argv[1]
except IndexError:
    print("IndexError: Please provide a sub-folder name in the command line argument")
    exit()

# create the dataset folders
path = os.path.join(datasets, subdata)
if not os.path.isdir(path):
    os.mkdir(path)

(height, width) = (100, 130)  # set the image size

# haar cascade - create a cascade initialized with the face cascade
# Loads the face cascade (XML file containing data to detect faces) into memory
faceCascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # set video source as default webcam

# loop the program until we have 30 images of the user face
img_count = 1
while img_count < 30:
    (_, image) = webcam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 4)

    # draw a identifying rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (height, width))
        cv2.imwrite('% s/% s.png' % (path, img_count), face_resize)

    img_count += 1

    cv2.imshow('Generator', image)
    key = cv2.waitKey(10)
    if key == 27:
        break
