from cv2 import cv2
import sys

# paths to the image and cascade names passed in as command-line arguments.
imgPath = sys.argv[1]

# haar cascade - create a cascade initialized with the face cascade
# Loads the face cascade (XML file containing data to detect faces) into memory
faceCascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_default.xml')

# read the given image and convert it to greyscale
image = cv2.imread(imgPath)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# FACE DETECTION

# detectMultiScale detects objects, called on a face cascade
# this will return a list of rectangles inside which a face has been detected
faces = faceCascade.detectMultiScale(
    grey,  # greyscale image
    scaleFactor=1.1,  # compensates for the different distances from the camera of the faces
    minNeighbors=5,  # defines how many objects are detected near the current object before declaring a face is detected
    minSize=(30, 30),  # size of each window
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces.".format(len(faces)))

# draw a identifying rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Found", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
