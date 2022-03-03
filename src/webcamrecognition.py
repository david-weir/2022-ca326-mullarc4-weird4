import face_recognition
import pickle
import cv2
from datetime import datetime
import numpy as np
from keras.models import load_model


def live_attendance():

    model = load_model("./mask_detector.model")  # load trained model

    # labels and colour dictionaries
    labels_dict = {0: 'Mask', 1: 'No Mask'}
    colour_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

    size = 4

    # load the haar cascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_frontalface_alt2.xml')
    # load the known faces and embeddings
    data = pickle.loads(open('face_enc', "rb").read())

    print("Streaming...")

    # begin webcam feed
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # main camera capture at 0

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

            # update the list of names
            names.append(name)

            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):

                # rescale the face coordinates

                # Save just the rectangle faces in SubRecFaces
                # resize images
                face_img = frame[y:y + h, x:x + w]
                resized = cv2.resize(face_img, (224, 224))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 224, 224, 3))
                reshaped = np.vstack([reshaped])
                result = model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]  # mask or no mask label prediction

                # take attendance info for recognised student, append it to attendance information list
                if name not in seen:
                    student = []
                    student.append(name)
                    student.append(datetime.now().strftime("%H:%M:%S"))
                    student.append(labels_dict[label])

                    attendance.append(student)
                    seen.append(name)  # update list of seen students

                # mark the detected face with a rectangle drawn on the display
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour_dict[label], 2)

                # put the name of the recognised person/face on the screen
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Mask / No Mask Label
                cv2.putText(frame, labels_dict[label], (x + 5, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 1,
                            colour_dict[label], 2)

        cv2.imshow("Webcam", frame)  # display

        # end program if "q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # create a rollbook text file with today's data
    rollbook_name = "RollBook " + datetime.today().strftime("%d.%m.%y") + ".csv"
    rollbook = open(rollbook_name, "w")

    # rollbook titles
    rollbook.write("Student Name, " + "Time, " + "Mask/No Mask\n")

    # add attendance info for each student to a comma delimited text file
    for info in attendance:
        rollbook.write(info[0] + ", " + info[1] + ", " + info[2] + "\n")

    webcam.release()
    cv2.destroyAllWindows()
