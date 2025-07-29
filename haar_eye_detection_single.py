import cv2
import matplotlib.pyplot as plt

# Load pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect faces and eyes
def detect_faces_and_eyes(image):
    if image is None:
        print("Error: Unable to read the image.")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return image

# Test on an image
image_path = "C:/Users/PMLS/Downloads/MV5BYzI5MjBkMWQtMjE1My00NWJlLTg4MWQtY2FlNmUxNjUwZDQ3XkEyXkFqcGc@._V1_.jpg"# Replace with your image path
image = cv2.imread(image_path)

# Detect faces and eyes
result_image = detect_faces_and_eyes(image)

# Display the result in Jupyter Notebook using Matplotlib
if result_image is not None:
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
    plt.figure(figsize=(8, 6))
    plt.imshow(result_image)
    plt.axis('off')  # Hide axis
    plt.show()
