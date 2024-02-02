import dlib
import cv2
import numpy as np

def detect_and_extract_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Initialize the face detector from dlib
    # face_detector = dlib.get_frontal_face_detector()
    face_detector = dlib.cnn_face_detection_model_v1("trained_models/mmod_human_face_detector.dat")
    
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    face = face_detector(gray_image)

    if face:
        # Iterate over detected rectangles
        for mmod_rect in face:
            # Convert the mmod_rectangle to rectangle
            rect = dlib.rectangle(mmod_rect.rect.left(), mmod_rect.rect.top(),
                                mmod_rect.rect.right(), mmod_rect.rect.bottom())

        return rect
    else:
        print("No faces found in one or both images.")
        return "No faces Found"



# def main():
#     image_path = 'passport.jpeg'
    
#     # Detect and extract faces
#     extracted_faces = detect_and_extract_faces(image_path)
    
#     print(extracted_faces)


# Load the pre-trained face recognition model from dlib
face_rec_model = dlib.face_recognition_model_v1("trained_models/dlib_face_recognition_resnet_model_v1.dat")

# Load the facial landmark predictor from dlib
shape_predictor = dlib.shape_predictor("trained_models/shape_predictor_68_face_landmarks.dat")


def extract_face_features(image, face):
     # Detect facial landmarks
    landmarks = shape_predictor(image, face)
    
    # Convert image to RGB (dlib uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Compute face descriptor (embedding)
    face_descriptor = face_rec_model.compute_face_descriptor(image_rgb, landmarks)
    
    return face_descriptor


def compare_faces(face_descriptor1, face_descriptor2):
    # Calculate Euclidean distance between the two face descriptors
    distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))
    
    return distance


def main():
    image_path_1 = 'passport.jpeg'
    image_path_2 = 'passport3.jpeg'
    
    # Load images
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    #extract faces from images
    faces1 = detect_and_extract_faces(image_path_1)
    faces2 = detect_and_extract_faces(image_path_2)
    
    # if len(faces1) == 0 or len(faces2) == 0:
    #     print("No faces found in one or both images.")
    #     return
    
    # Extract features from the first face in each image
    face_descriptor1 = extract_face_features(image1, faces1)
    face_descriptor2 = extract_face_features(image2, faces2)
    
    # Compare face descriptors
    distance = compare_faces(face_descriptor1, face_descriptor2)
    
    print("Euclidean distance between the faces:", distance)


if __name__ == "__main__":
    main()
