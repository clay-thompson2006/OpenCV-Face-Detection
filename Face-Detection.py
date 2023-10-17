import cv2
import face_recognition

# Load images and create face encodings
hayden_image = face_recognition.load_image_file("Hayden.jpg")
hayden_face_encoding = face_recognition.face_encodings(hayden_image)[0]

sekol_image = face_recognition.load_image_file("Sekol.jpg")
sekol_face_encoding = face_recognition.face_encodings(sekol_image)[0]

known_face_encodings = [
    hayden_face_encoding,
    sekol_face_encoding
]
known_face_names = [
    "Hayden Hetrick",
    "Mr. Sekol"
]

# Initialize video capture from the default camera (change the argument to use a different camera)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Find face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare the current face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
