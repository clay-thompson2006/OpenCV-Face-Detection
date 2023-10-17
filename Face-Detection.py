import cv2 # Import OpenCV Library

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Initialize the video capture object to capture video from the default camera.
video_capture = cv2.VideoCapture(0)
# Define a function to detect faces and draw bounding boxes around them in the input video frame.
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)  # Use cv2.COLOR_BGR2GRAY for grayscale conversion
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x,y), (x + w, y + h), (0, 255, 0), 4)
    return faces    
# Start an infinite loop for video capture and processing.
while True:

    result, video_frame = video_capture.read()  # reads frames from the video
    if result is False:
        break  # terminates the loop if the frame is not read successfully
    
    faces = detect_bounding_box(video_frame)  # applies the function created to the video frame

    cv2.imshow("Haydens Face Detection Project", video_frame)  # displays the processed frame in a window named "Haydens Face Detection Project"

    # Check for the 'q' key press to exit the loop and terminate the program.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release the video capture object and close all OpenCV windows.
video_capture.release()
cv2.destroyAllWindows()
