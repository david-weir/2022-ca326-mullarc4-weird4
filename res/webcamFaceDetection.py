from cv2 import cv2

# haar cascade - create a cascade initialized with the face cascade
# Loads the face cascade (XML file containing data to detect faces) into memory
faceCascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # set video source as default webcam

# FACE DETECTION

while True:

    ret, frame = webcam.read()  # capture webcam feed frame-by-frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,  # greyscale image
        scaleFactor=1.5,  # compensates for the different distances from the camera of the faces
        minNeighbors=5,  # defines how many objects are detected near the current object before
                         # declaring a face is detected
        minSize=(30, 30),  # size of each window
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # draw a identifying rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces Found", frame)

    # if q is pressed -> exit script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
