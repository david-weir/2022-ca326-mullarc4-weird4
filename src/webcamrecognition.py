import face_recognition
import pickle
import cv2
from datetime import datetime

# load the haar cascade in the cascade classifier
faceCascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_alt2.xml')
# load the known faces and embeddings
data = pickle.loads(open('face_enc', "rb").read())

print("Streaming...")

# begin webcam feed
webcam = cv2.VideoCapture(0)  # main camera capture at 0

seen = []
attendance = []  # list of student attendance and times

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,  # greyscale image
        scaleFactor=1.1,  # compensates for the different distances from the camera of the faces
        minNeighbors=5,  # defines how many objects are detected near the current object before
        # declaring a face is detected
        minSize=(60, 60),  # size of each window
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # the facial embeddings for the face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []

    # loop over the facial embeddings
    # this loops over the embeddings of each face we have in the datasets folder
    # it allow us to recognise all the faces on the web feed (assuming there is more than 1 at any time)
    for encoding in encodings:

        # Compare encodings with encodings in data["encodings"]

        # matches will be an array of True boolean values for images that match the data
        # and False boolean values for images that are unknown or do not match
        matches = face_recognition.compare_faces(data["encodings"], encoding)

        # set name as unknown if no encoding matches exist
        name = "Unknown"

        # checks for found matches
        if True in matches:

            # Find positions at which we get True and store
            matchedIds = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIds:

                # Check the names at respective indexes we stored in matchedIds
                name = data["names"][i]

                # increase count for that name
                counts[name] = counts.get(name, 0) + 1

            # set the name which has highest count
            name = max(counts, key=counts.get)

        if name not in seen:
            student = []
            student.append(name)
            student.append(datetime.now().strftime("%H:%M:%S"))

            attendance.append(student)

        # update the list of names
        names.append(name)
        seen.append(name)

        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):

            # rescale the face coordinates

            # mark the detected face with a rectangle drawn on the display
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # put the name of the recognised person/face on the screen
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

rollbook_name = "RollBook " + datetime.today().strftime("%d.%m.%y") + ".txt"
rollbook = open(rollbook_name, "w")

for info in attendance:
    rollbook.write(info[0] + ", " + info[1] + "\n")

webcam.release()
cv2.destroyAllWindows()
