# 🕵️‍♂️ Object Detection using Haar Cascades

🎯 Objective  
To implement face and eye detection using pre-trained Haar-feature classifiers with OpenCV. This lab includes:

- Detecting faces and eyes in a single static image  
- Processing multiple images to evaluate detection accuracy  

📁 Code Files  
🔹 haar_eye_detection_single.py  
This script performs face and eye detection on a single image using Haar cascades.

📌 Key Components:  
Classifiers: haarcascade_frontalface_default.xml, haarcascade_eye.xml  
Function: detect_faces_and_eyes()  
Technique: Convert image to grayscale, detect face, then detect eyes inside face region  
Visualization: Uses Matplotlib to display final output  

📈 Output:  
- Blue rectangles drawn around detected faces  
- Green rectangles drawn around detected eyes  
- Output image displayed using matplotlib.pyplot  

🔹 haar_eye_detection_batch.py  
This script processes the first 10 images from a folder, detects faces and eyes, and calculates detection accuracy.

📌 Key Components:  
Input: Loads .jpg, .jpeg, or .png files from a specified folder  
Loop: Detects and draws bounding boxes for face and eyes for each image  
Evaluation: Assumes 1 face per image to calculate detection accuracy  

📈 Output:  
- Displays 10 images in a 2x5 subplot format with bounding boxes  
- Prints total faces detected and calculated accuracy percentage in terminal  

🧰 Requirements  
Install dependencies using:

```bash
pip install opencv-python matplotlib
🔁 How to Run

For single image detection:python haar_eye_detection_single.py
For batch image detection (10 images):python haar_eye_detection_batch.py
✅ Ensure correct paths are set in the scripts:

image_path in haar_eye_detection_single.py

image_folder in haar_eye_detection_batch.py📌 Notes

Haar Cascades are efficient and suitable for real-time detection

May result in false positives in complex scenes

Accuracy depends on image quality, lighting, and visibility

For advanced object detection, consider deep learning methods like YOLO or SSD

