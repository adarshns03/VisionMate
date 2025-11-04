import cv2
from ultralytics import YOLO
import pytesseract
import pyttsx3
import requests
import numpy as np
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import tkinter as tk
from tkinter import filedialog
import time
from deepface import DeepFace

# TTS setup
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


# Constants and API setup
PAT = ' .. '
USER_ID = 'clarifai'
APP_ID = 'main'
MODEL_ID = 'celebrity-face-recognition'
MODEL_VERSION_ID = '0676ebddd5d6413ebdaa101570295a39'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
object_model = YOLO("yolov8n.pt")  # Renamed for clarity

# Distance tracking constants
KNOWN_WIDTHS = {
    "cell phone": 7.0, "cup": 8.0, "book": 15.0,
    "bottle": 7.5, "laptop": 30.0, "keyboard": 35.0, "mouse": 10.0
}
DEFAULT_WIDTH = 10.0
FOCAL_LENGTH = 1000

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', 'Key ' + PAT),)
userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)


# Utility functions
def resize_to_window(img, width=800, height=600):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    return cv2.resize(img, (int(w * scale), (int(h * scale))))


def run_object_detection(img):
    results = object_model(img)
    annotated = results[0].plot()
    names = [object_model.names[int(cls)] for cls in results[0].boxes.cls]
    unique = set(names)

    print("\nDetected Objects:")
    print(*unique or ["None"], sep="\n- ")
    speak("Detected objects are: " + ", ".join(unique) if unique else "No objects detected.")
    return annotated


def run_text_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
    print("\nDetected Text:\n" + (text or "No text detected."))
    speak("Detected text is: " + text if text else "No text detected.")
    return thresh


def run_celebrity_detection(img):
    # Resize image for optimal face detection (500x500 for single faces)
    h, w = img.shape[:2]

    # Calculate resize ratio (maintain aspect ratio)
    target_size = 500
    ratio = target_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to bytes for Clarifai API
    _, img_encoded = cv2.imencode('.jpg', resized_img)

    response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=img_encoded.tobytes())
                    )
                )
            ]
        ), metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        print("Clarifai request failed.")
        speak("Failed to recognize celebrity")
        return img

    concepts = response.outputs[0].data.concepts
    if concepts:
        name, confidence = concepts[0].name, concepts[0].value
        print(f"\nCelebrity: {name} ({confidence * 100:.2f}%)")
        speak(f"The most likely celebrity is {name} with {confidence * 100:.2f} percent confidence")
        annotated = img.copy()
        cv2.putText(annotated, f"{name} ({confidence * 100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated
    else:
        print("No celebrity recognized.")
        speak("No celebrity recognized")
        return img


# New function for emotion detection
def run_emotion_detection():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        speak("Could not open webcam")
        return_to_base()
        return

    last_emotion = None
    last_time = time.time()
    emotion_interval = 7  # seconds

    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if current_time - last_time >= emotion_interval or last_emotion is None:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if results and isinstance(results, list):
                    last_emotion = results[0]['dominant_emotion']
                    last_time = current_time
                    speak(f"You look {last_emotion}")
            except Exception as e:
                print(f"Emotion detection error: {str(e)}")
                last_emotion = "unknown"

        display_text = f'Emotion: {last_emotion or "Detecting..."}'
        cv2.putText(frame, display_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyWindow("Emotion Detection")
    return_to_base()


# New function for object distance tracking
def run_distance_tracking():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        speak("Could not open webcam")
        return_to_base()
        return

    last_spoken = {}
    cv2.namedWindow("Object Distance Tracker", cv2.WINDOW_NORMAL)

    def calc_distance(p_width, a_width):
        return (a_width * FOCAL_LENGTH) / p_width if p_width else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = object_model(frame)
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                      r.boxes.cls.cpu().numpy(),
                                      r.boxes.conf.cpu().numpy()):
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box)
                name = object_model.names[int(cls)]
                width_px = x2 - x1
                width_cm = KNOWN_WIDTHS.get(name, DEFAULT_WIDTH)
                dist = calc_distance(width_px, width_cm)

                # Only speak if significant distance change or first detection
                prev = last_spoken.get(name, 0)
                if abs(prev - dist) > 10 or name not in last_spoken:
                    speak(f"{name} is {int(dist)} centimeters away")
                    last_spoken[name] = dist

                label = f"{name}: {dist:.1f}cm"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Object Distance Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Object Distance Tracker")
    return_to_base()


# Helper to return to base image
def return_to_base():
    cv2.namedWindow("Base Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Base Image", base_image)
    cv2.resizeWindow("Base Image", 800, 600)


# Function to open image dialog
def open_image_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
    root.destroy()
    if file_path:
        img = cv2.imread(file_path)
        return img, file_path
    else:
        return None, None


# Image paths (initial defaults)
base_img_path = r"C:\Users\adars\OneDrive\Documents\Blind\bg.png"

# Load default images
base_image = resize_to_window(cv2.imread(base_img_path))

current_image = None
current_image_path = None

# Show only base image at startup
cv2.namedWindow("Base Image", cv2.WINDOW_NORMAL)
cv2.imshow("Base Image", base_image)
cv2.resizeWindow("Base Image", 800, 600)

print("Press:")
print("'p' - Detect Objects")
print("'t' - Detect Text")
print("'c' - Recognize Celebrity")
print("'e' - Emotion Detection (Webcam)")
print("'d' - Object Distance Tracking (Webcam)")  # New option
print("'o' - Open Image for processing")
print("'q' - Quit")

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord('o'):  # Upload image
        img, path = open_image_dialog()
        if img is not None:
            current_image = img
            current_image_path = path
            # Resize uploaded image to 500x700 directly
            resized = cv2.resize(current_image, (500, 700), interpolation=cv2.INTER_AREA)
            cv2.namedWindow("Loaded Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Loaded Image", resized)
        else:
            print("No image selected.")
            speak("No image selected.")

    elif key in [ord('p'), ord('t'), ord('c')]:  # Require uploaded image
        if current_image is None:
            print("Please upload an image first (press 'o').")
            speak("No image loaded. Press O to upload one.")
            continue

        if key == ord('p'):  # Object detection
            result = resize_to_window(run_object_detection(current_image))
            cv2.imshow("Loaded Image", result)

        elif key == ord('t'):  # Text detection
            result = resize_to_window(run_text_detection(current_image))
            cv2.imshow("Loaded Image", result)

        elif key == ord('c'):  # Celebrity detection
            result = run_celebrity_detection(current_image)
            # Resize the celebrity detection output to 500x700
            resized_result = cv2.resize(result, (500, 700), interpolation=cv2.INTER_AREA)
            cv2.imshow("Loaded Image", resized_result)

    elif key in [ord('e'), ord('d')]:  # Webcam functions (no image needed)
        if key == ord('e'):
            run_emotion_detection()
        elif key == ord('d'):
            run_distance_tracking()

    elif key == ord('q'):  # Quit
        break


cv2.destroyAllWindows()
