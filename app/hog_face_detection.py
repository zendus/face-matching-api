import os
import dlib
import cv2
import numpy as np


# Define the path to the trained model file
models_dir = "trained_models"
face_rec_model_file_path = os.path.join(os.path.dirname(__file__), models_dir, "dlib_face_recognition_resnet_model_v1.dat")
shape_predictor_model_file_path = os.path.join(os.path.dirname(__file__), models_dir, "shape_predictor_68_face_landmarks.dat")

# Check if the model file exists
if not os.path.exists(face_rec_model_file_path):
    raise FileNotFoundError(f"Model file '{face_rec_model_file_path}' not found.")
elif not os.path.exists(shape_predictor_model_file_path):
    raise FileNotFoundError(f"Model file '{shape_predictor_model_file_path}' not found.")

# Load the face recognition model
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_file_path)

# Load the facial landmark predictor from dlib
shape_predictor = dlib.shape_predictor(shape_predictor_model_file_path)


async def detect_and_extract_faces(gray_image):
    
    # Initialize the face detector from dlib
    face_detector = dlib.get_frontal_face_detector()
    
    # Detect faces in the image
    face = face_detector(gray_image)

    return face



def extract_face_features(image, face):
     # Detect facial landmarks
    landmarks = shape_predictor(image, face)
    
    # Convert landmarks to a numpy array
    landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # Convert image to RGB (dlib uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Compute face descriptor (embedding)
    face_descriptor = face_rec_model.compute_face_descriptor(image_rgb, landmarks)
    
    return face_descriptor


def compare_faces(face_descriptor1, face_descriptor2):
    # Calculate Euclidean distance between the two face descriptors
    distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))
    
    return distance
