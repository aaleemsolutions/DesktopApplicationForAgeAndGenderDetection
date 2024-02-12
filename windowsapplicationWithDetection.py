import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import matplotlib.pyplot as plt


# Function to get face bounding boxes from a frame
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Function to detect age and gender from a frame
def age_gender_detector(frame, faceNet, ageNet, genderNet):
    padding = 20
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frameFace

# Load pre-trained models
faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"
ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"
genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define gender and age lists
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# Function to update the displayed image
def update_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(img)

    # Get the current window size
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    # Calculate the target width and height based on the window size
    target_width = int(0.7 * window_width)
    target_height = int(0.7 * window_height)

    # Resize the image without antialiasing
    img_pil_resized = img_pil.resize((target_width, target_height))

    # Convert PIL Image to PhotoImage
    img_tk = ImageTk.PhotoImage(image=img_pil_resized)

    # Update the label with the new image
    label.config(image=img_tk)
    label.image = img_tk

# Function to handle selecting images
def select_images():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Call the age_gender_detector function
        output_image = age_gender_detector(img, faceNet, ageNet, genderNet)

        # Update the displayed image
        update_image(output_image)

# Create the main window
window = tk.Tk()
window.title("BC200200019 - Age and Gender Detection Final Year Project")



    
# Get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set window size (70% of screen width and height)
window_width = int(0.7 * screen_width)
window_height = int(0.7 * screen_height)
window.geometry(f"{window_width}x{window_height}")

# Center the window
window_x = int((screen_width - window_width) / 2)
window_y = int((screen_height - window_height) / 2)
window.geometry(f"+{window_x}+{window_y}")


# Create buttons and labels
select_images_button = tk.Button(window, text="Select Images", command=select_images)
select_images_button.pack()

label = tk.Label(window)
label.pack()

# Run the Tkinter event loop
if __name__ == "__main__":
    window.mainloop()
