# How to run?: python stream.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python stream.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import time
import screeninfo
import sys
import collections
from datetime import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# Arguments used here:
# prototxt = MobileNetSSD_deploy.prototxt.txt (required)
# model = MobileNetSSD_deploy.caffemodel (required)
# confidence = 0.2 (default)

# SSD (Single Shot MultiBox Detector) is a popular algorithm in object detection
# It has no delegated region proposal network and predicts the boundary boxes and the classes directly from feature maps in one single pass
# To improve accuracy, SSD introduces: small convolutional filters to predict object classes and offsets to default boundary boxes
# Mobilenet is a convolution neural network used to produce high-level features

# SSD is designed for object detection in real-time
# The SSD object detection composes of 2 parts: Extract feature maps, and apply convolution filters to detect objects

# Let's start by initialising the list of the 21 class labels MobileNet SSD was trained to.
# Each prediction composes of a boundary box and 21 scores for each class (one extra class for no object),
# and we pick the highest score as the class for the bounded object
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

CORRECT_ELEMENTS = ["cat", "dog"]
CORRECT_INDICES = [CLASSES.index(i) for i in CORRECT_ELEMENTS]

# Assigning random colors to each of the classes
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Assign color green to "correct" element, color red to everything else
# Note: The color order is weird and is BGR instead of RGB
red_bgr = [0.000000000, 0.000000000, 255.000000000]
green_bgr = [0.000000000, 128.000000000, 0.000000000]
gray_bgr = [211., 211., 211.]
COLORS = [red_bgr] * (len(CLASSES) - 1)
for i in CORRECT_INDICES:
	COLORS.insert(i, green_bgr)
COLORS = np.asarray(COLORS)


# COLORS: a list of 21 R,G,B values, like ['101.097383   172.34857188 111.84805346'] for each label
# length of COLORS = length of CLASSES = 21

# load our serialized model
# The model from Caffe: MobileNetSSD_deploy.prototxt.txt; MobileNetSSD_deploy.caffemodel;
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# print(net)
# <dnn_Net 0x128ce1310>

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# warm up the camera for a couple of seconds
time.sleep(2.0)

# FPS: used to compute the (approximate) frames per second
# Start the FPS timer
fps = FPS().start()

# OpenCV provides two functions to facilitate image preprocessing for deep learning classification: cv2.dnn.blobFromImage and cv2.dnn.blobFromImages. Here we will use cv2.dnn.blobFromImage
# These two functions perform: Mean subtraction, Scaling, and optionally channel swapping

# Mean subtraction is used to help combat illumination changes in the input images in our dataset. We can therefore view mean subtraction as a technique used to aid our Convolutional Neural Networks
# Before we even begin training our deep neural network, we first compute the average pixel intensity across all images in the training set for each of the Red, Green, and Blue channels.
# we end up with three variables: mu_R, mu_G, and mu_B (3-tuple consisting of the mean of the Red, Green, and Blue channels)
# For example, the mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68
# When we are ready to pass an image through our network (whether for training or testing), we subtract the mean, \mu, from each input channel of the input image:
# R = R - mu_R
# G = G - mu_G
# B = B - mu_B

# We may also have a scaling factor, \sigma, which adds in a normalization:
# R = (R - mu_R) / sigma
# G = (G - mu_G) / sigma
# B = (B - mu_B) / sigma

# The value of \sigma may be the standard deviation across the training set (thereby turning the preprocessing step into a standard score/z-score)
# sigma may also be manually set (versus calculated) to scale the input image space into a particular range — it really depends on the architecture, how the network was trained

# cv2.dnn.blobFromImage creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels
# a blob is just an image(s) with the same spatial dimensions (width and height), same depth (number of channels), that have all be preprocessed in the same manner

# Consider the video stream as a series of frames. We capture each frame based on a certain FPS, and loop over each frame
# loop over the frames from the video stream

screen = screeninfo.get_monitors()[0]
screen_w, screen_h = screen.width, screen.height

found_elements = False

prev_box_locations = collections.defaultdict(list)
new_box_locations = collections.defaultdict(list)

red_dot_counter = 0

while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	# vs is the VideoStream
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# print(frame.shape) # (225, 400, 3)
	# grab the frame dimensions and convert it to a blob
	# First 2 values are the h and w of the frame. Here h = 225 and w = 400
	(h, w) = frame.shape[:2]
	# Resize each frame
	resized_image = cv2.resize(frame, (300, 300))
	# Creating the blob
	# The function:
	# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
	# image: the input image we want to preprocess before passing it through our deep neural network for classification
	# mean:
	# scalefactor: After we perform mean subtraction we can optionally scale our images by some factor. Default = 1.0
	# scalefactor  should be 1/sigma as we're actually multiplying the input channels (after mean subtraction) by scalefactor (Here 1/127.5)
	# swapRB : OpenCV assumes images are in BGR channel order; however, the 'mean' value assumes we are using RGB order.
	# To resolve this discrepancy we can swap the R and B channels in image  by setting this value to 'True'
	# By default OpenCV performs this channel swapping for us.

	blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
	# print(blob.shape) # (1, 3, 300, 300)
	# pass the blob through the network and obtain the predictions and predictions
	net.setInput(blob) # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	# Predictions:
	predictions = net.forward()

	correct_idx_set = set(CORRECT_INDICES)

	frame = imutils.resize(frame, width=screen_w*2, height=screen_h*2)
	(h, w) = frame.shape[:2]
	frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)

	# loop over the predictions
	for i in np.arange(0, predictions.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		# predictions.shape[2] = 100 here
		confidence = predictions[0, 0, i, 2]
		# Filter out predictions lesser than the minimum confidence level
		# Here, we set the default confidence as 0.2. Anything lesser than 0.2 will be filtered
		if confidence > args["confidence"]:
			# extract the index of the class label from the 'predictions'
			# idx is the index of the class label
			# E.g. for person, idx = 15, for chair, idx = 9, etc.
			idx = int(predictions[0, 0, i, 1])
			category = CLASSES[idx]

			# then compute the (x, y)-coordinates of the bounding box for the object
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			# Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
			# Convert them to integers: 130 76 393 224
			(startX, startY, endX, endY) = box.astype("int")

			if category in prev_box_locations:
				closest = [sys.maxsize] * 4
				closest_eucl = (float('inf'), float('inf'))
				found = False
				for coord in prev_box_locations[category]:
					start_prev = np.array([coord[:2]])
					end_prev = np.array(coord[2:])
					start_curr = np.array([startX, startY])
					end_curr = np.array([endX, endY])

					eucl_start = np.linalg.norm(start_prev - start_curr)
					eucl_end = np.linalg.norm(end_prev - end_curr)
					if eucl_start < 150 and eucl_end < 150 and eucl_start < closest_eucl[0] and eucl_end < closest_eucl[1]:
						closest = coord
						closest_eucl = (eucl_start, eucl_end)
						found = True

				if found:
					(startX, startY, endX, endY) = closest
					new_box_locations[category].append(closest)
				else:
					new_box_locations[category].append((startX, startY, endX, endY))
			else:
				new_box_locations[category].append((startX, startY, endX, endY))

			# Get the label with the confidence score
			label = "{}: {:.0f}%".format(category, confidence * 100)
			print("Object detected: ", label)
			# Draw a rectangle across the boundary of the object
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 10)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			# Put a text outside the rectangular detection
			# Choose the font of your choice: FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC, etc.
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 10)

			if idx in correct_idx_set:
				correct_idx_set.remove(idx)

			if len(correct_idx_set) == 0:
				found_elements = True

	prev_box_locations = new_box_locations
	new_box_locations = collections.defaultdict(list)

	margin = 50
	thickness = 35
	length = 400

	# Top left
	cv2.rectangle(frame, (margin, margin), (margin + thickness, margin + length), gray_bgr, -1)
	cv2.rectangle(frame, (margin, margin), (margin + length, margin + thickness), gray_bgr, -1)

	# Top right
	cv2.rectangle(frame, (w - margin - thickness, margin), (w - margin, margin + length), gray_bgr, -1)
	cv2.rectangle(frame, (w - margin - length, margin), (w - margin, margin + thickness), gray_bgr, -1)

	# Bottom left
	cv2.rectangle(frame, (margin, h - margin - length), (margin + thickness, h - margin), gray_bgr, -1)
	cv2.rectangle(frame, (margin, h - margin - thickness), (margin + length, h - margin), gray_bgr, -1)

	# Bottom right
	cv2.rectangle(frame, (w - margin - thickness, h - margin - length), (w - margin, h - margin), gray_bgr, -1)
	cv2.rectangle(frame, (w - margin - length, h - margin - thickness), (w - margin, h - margin), gray_bgr, -1)

	cv2.putText(frame, "CAM 03", (margin + thickness + 20, h - margin - thickness - 20), cv2.FONT_HERSHEY_DUPLEX, 3, [0,0,0], 7)
	cv2.putText(frame, "REC", (w//2 - 100, margin + thickness + 50), cv2.FONT_HERSHEY_DUPLEX, 4, [0,0,0], 6)

	if red_dot_counter < 10:
		cv2.circle(frame, (w//2 + 200, margin +thickness + 10), 25, red_bgr, -1)
	elif red_dot_counter == 20:
		red_dot_counter = 0
	red_dot_counter += 1

	cv2.rectangle(frame, (w//2  - 225, h - margin - thickness - 140), (w // 2 + 265, h - margin - 50), gray_bgr, -1)

	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	cv2.putText(frame, current_time, (w//2 - 225, h - margin - thickness - 25), cv2.FONT_HERSHEY_PLAIN, 7, [0,0,0], 9)

	# show the output frame'
	frame = imutils.resize(frame, width=screen_w*2, height=screen_h*2)

	name = 'Security Camera'
	cv2.namedWindow(name)
	# cv2.moveWindow(name, screen.x, screen.y)
	cv2.imshow(name, frame)
	
	# HOW TO STOP THE VIDEOSTREAM?
	# Using cv2.waitKey(1) & 0xFF

	# The waitKey(0) function returns -1 when no input is made
	# As soon an event occurs i.e. when a button is pressed, it returns a 32-bit integer

	# since we only require 8 bits to represent a character we AND waitKey(0) to 0xFF, an integer below 255 is always obtained
	# ord(char) returns the ASCII value of the character which would be again maximum 255
	# by comparing the integer to the ord(char) value, we can check for a key pressed event and break the loop
	# ord("q") is 113. So once 'q' is pressed, we can write the code to break the loop
	# Case 1: When no button is pressed: cv2.waitKey(1) is -1; 0xFF = 255; So -1 & 255 gives 255
	# Case 2: When 'q' is pressed: ord("q") is 113; 0xFF = 255; So 113 & 255 gives 113

	# So we will basically get the ord() of the key we press if we do a bitwise AND with 255.
	# ord() returns the unicode code point of the character. For e.g., ord('a') = 97; ord('q') = 113

	# Now, let's code this logic (just 3 lines, lol)
	key = cv2.waitKey(1) & 0xFF

	# Press 'q' key to break the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

	# Exit if element found
	if found_elements == True:
		print("YAY! you found the elements")
		print("Your next clue is <Insert clue here>")
		time.sleep(10)
		break

# stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()

# In case you removed the opaque tape over your laptop cam, make sure you put them back on once finished ;)
# YAYYYYYYYYYY WE ARE DONE!
