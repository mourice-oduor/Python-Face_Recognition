import cv2
import numpy as np
import os
import face_recognition
import face_recognition as fr

from time import sleep


def encodingFaces():
    encoded = {}      ##return: dict of (name, encoded_image)

    for dirpath, dnames, fnames in os.walk(".Input_/faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("Input_faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def encodedImage(img):
    face = fr.load_image_file("Input_faces/" + img)
    encoding = fr.face_encodings(face)[0]
    return encoding


def faceClassification(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = encodingFaces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    image = cv2.imread(im, 1)
    #image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    #image = image[:,:,::-1]
 
    face_locations = face_recognition.face_locations(image)
    unknown_face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(image, (left-20, top-20), (right+20, bottom+20), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left-20, bottom -15), (right+20, bottom+20), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Recognition', image)
        #cv2.imwrite('/home/net/MORYSO/PYTHON/Game-Dev/games-todo/14. Face-Recognition/Face-Recognition/Recognized_images/recognized_image{}.jpg'.format(image), image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            return face_names

        cv2.destroyAllWindows()

if __name__ == "__main__":
    print(faceClassification("test.jpg"))

