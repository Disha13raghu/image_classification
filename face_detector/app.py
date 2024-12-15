import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces():
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam.")
        return

    print("Starting face detection. Press 'q' to quit.")

    while True:
        success, frame = webcam.read()

        if not success:
            print("Failed to grab frame from webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection in Action", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting the detection loop.")
            break

    webcam.release()
    cv2.destroyAllWindows()
    print("Resources released successfully.")


if __name__ == "__main__":
    detect_faces()
