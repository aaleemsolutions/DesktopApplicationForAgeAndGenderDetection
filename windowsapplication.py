import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

cap = None  # Global variable to hold the camera object

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

def select_images():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        label.config(image=img)
        label.image = img

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        label.config(image=img)
        label.image = img
        if cap.isOpened():  # Check if the camera is still open
            window.after(10, update_frame)

def close_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        label.config(image='')  # Clear the label when closing the camera

def close_window():
    close_camera()
    window.destroy()

window = tk.Tk()
window.title("FYP Project")

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

button_frame = tk.Frame(window)
button_frame.pack(side=tk.TOP)

open_camera_button = tk.Button(button_frame, text="Open Camera", command=open_camera)
open_camera_button.pack(side=tk.LEFT, padx=5, pady=5)

close_camera_button = tk.Button(button_frame, text="Close Camera", command=close_camera)
close_camera_button.pack(side=tk.LEFT, padx=5, pady=5)

close_window_button = tk.Button(button_frame, text="Close Window", command=close_window)
close_window_button.pack(side=tk.LEFT, padx=5, pady=5)

select_images_button = tk.Button(button_frame, text="Select Images", command=select_images)
select_images_button.pack(side=tk.LEFT, padx=5, pady=5)





frame = tk.Frame(window)
frame.pack()

label = tk.Label(frame)
label.pack()

window.protocol("WM_DELETE_WINDOW", close_window)
window.mainloop()
