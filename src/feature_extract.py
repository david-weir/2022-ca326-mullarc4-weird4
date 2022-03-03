# Train face recognition model
# Extracts facial features from the images in each sub-folder in the datasets directory

import os, cv2, pickle, face_recognition
from imutils import paths

def extract():
    # retrieve the path of each sub-data folder in the datasets folder
    # the datasets folder contains sub-folders containing images of various people
    imgPaths = list(paths.list_images('datasets'))

    names, knownEncodings = [], []

    # loop through the image paths list
    for (i, imgPath) in enumerate(imgPaths):
        name = imgPath.split(os.path.sep)[-2]  # extract person's name from the image path

        # load the input image: convert it from OpenCV ordering BGR to dlib ordering RGB
        image = cv2.imread(imgPath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # locate faces using the face_recognition library
        boxes = face_recognition.face_locations(rgb, model='hog')

        # compute facial embeddings of the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over encodings list
        for encoding in encodings:
            knownEncodings.append(encoding)
            names.append(name)

    # save encodings and name in a data dictionary
    data = {
        "encodings": knownEncodings,
        "names": names
    }

    # save the data into a file for use later
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()
