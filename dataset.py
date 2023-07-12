
# face detection for the 5 Celebrity Faces Dataset and Kaggle Dataset

import os 
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

# extract the face from the image
def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = np.asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in os.listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in os.listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)

# load train dataset for 5_celebs
trainX, trainy = load_dataset('5_celebs/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('5_celebs/val/')
# save arrays to one file in compressed format
np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)


# load train dataset for Kaggle dataset
trainX, trainy = load_dataset('Kaggle_dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('Kaggle_dataset/val/')
# save arrays to one file in compressed format
np.savez_compressed('Kaggle_dataset.npz', trainX, trainy, testX, testy)


