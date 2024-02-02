from .hog_cnn_face_detection import detect_and_extract_faces_using_hog, detect_and_extract_faces_using_cnn, extract_face_features
from urllib.request import urlopen
from .error import FaceError
import numpy as np
import cv2



async def load_image_and_return_gray(cloudinary_url):
    response = urlopen(cloudinary_url)
    image_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    else:
        raise FaceError("Error: Failed to load the image from Cloudinary.")




async def generate_passport_descriptors(list_of_users):
	descriptors = {}
	for user in list_of_users:
		id = user.id
		image = await load_image_and_return_gray(user.passport_url)
		face = await detect_and_extract_faces_using_hog(image)
		face_descriptor = extract_face_features(image, face[0])
		descriptors[id] = face_descriptor
	return descriptors



def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)


async def get_one_face_descriptor(file):
    passport_image = await file.read()
    # Convert the bytes to a NumPy array representing the image
    nparr = np.frombuffer(passport_image, np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = await detect_and_extract_faces_using_hog(gray_image)

    face_descriptor = extract_face_features(image, face[0])

    return face_descriptor