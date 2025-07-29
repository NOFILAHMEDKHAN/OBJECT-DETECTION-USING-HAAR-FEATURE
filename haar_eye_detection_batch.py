import cv2
import os
import matplotlib.pyplot as plt

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define image folder path
image_folder = "C:/Users/PMLS/Documents/testimages/"  # Use forward slashes for Jupyter

# Get list of image files (JPG, PNG, JPEG)
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Select first 10 images (or fewer if less available)
image_files = image_files[:10]

# Function to detect faces and eyes
def detect_faces_and_eyes(image):
    if image is None:
        print(" Error: Unable to read the image.")
        return None, 0  # Return 0 detected faces

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    face_count = len(faces)  # Number of detected faces

    # Draw rectangles around faces and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green rectangle for eyes

    return image, face_count

# Process all selected images
total_detected_faces = 0
total_images = len(image_files)

plt.figure(figsize=(15, 10))  # Adjust figure size

for i, img_name in enumerate(image_files):
    img_path = os.path.join(image_folder, img_name)
    image = cv2.imread(img_path)

    # Detect faces and eyes
    result_image, detected_faces = detect_faces_and_eyes(image)
    total_detected_faces += detected_faces

    # Convert image to RGB for proper Matplotlib display
    if result_image is not None:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Show image
        plt.subplot(2, 5, i+1)
        plt.imshow(result_image)
        plt.title(f"Detected Faces: {detected_faces}")
        plt.axis("off")

# Calculate accuracy (if ground truth data is available)
accuracy = (total_detected_faces / (total_images * 1.0)) * 100  # Assuming 1 face per image
plt.show()

print(f" Object Detection Accuracy: {accuracy:.2f}%")
