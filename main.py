
import glob
import os
import time

import tensorflow as tf
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

from freezeGraph import *
from scipy.misc import imresize




if __name__ == '__main__':


	MODELS_PATH = 'models/'
	FREEZED_NAME = 'graph_freezed.pb'

	# Used in training
	#READ_PATH = '../Data/Clean_images/straightroad/'
	#READ_PATH = '../Data/Clean_images/startbox/'
	#READ_PATH = '../Data/Clean_images/roadnear/'
	#READ_PATH = '../Data/Clean_images/rightlanemissing/'
	#READ_PATH = '../Data/Clean_images/rightcurve/'
	#READ_PATH = '../Data/Clean_images/leftcurve/'
	READ_PATH = '../Data/Clean_images/fulltrack2/'
	#READ_PATH = '../Data/Clean_images/fulltrack1/'
	#READ_PATH = '../Data/Clean_images/dashedlinesmissing/'

	# Not used in training
	#READ_PATH = '../Data/Clean_images/obstacle/'

	# Missing annotation
	#READ_PATH = '../Data/Clean_images/intersectionobstacle/'
	#READ_PATH = '../Data/Clean_images/scurve/'

	# Not annotated
	#READ_PATH = '../Data/New_images/bosch/'
	#READ_PATH = '../Data/New_images/carolocup-left-outer/'
	#READ_PATH = '../Data/New_images/carolocup-right-inner-intersection/'
	#READ_PATH = '../Data/New_images/carolocup-right-outer/'
	#READ_PATH = '../Data/New_images/carolocup-right-outer-intersection/'

	AUTOPLAY = False
	CENTER_POINT_BOTTOM = (376, 480)
	IM_RESIZE_HEIGHT = 120
	IM_RESIZE_WIDTH = 180
	STACK_SIZE_TOP = 150
	STACK_SIZE_BOTTOM = 80

	# Read filenames from directory.
	images = sorted(glob.glob(READ_PATH + '*.png'), key=lambda name: int(name[len(READ_PATH):-4]))

	# Load labels if they exist.
	label_file_exist = os.path.isfile(READ_PATH + 'labels.csv')

	if label_file_exist:
		labels = pd.read_csv(READ_PATH + 'labels.csv', index_col='frame')

	with tf.Graph().as_default():
		
		# Load freezed graph from file.
		graph_def = tf.GraphDef()
		with open(MODELS_PATH + FREEZED_NAME, 'rb') as f:
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def)

		with tf.Session() as sess:

			# Load output node to use for predictions.
			output_node_processed = sess.graph.get_tensor_by_name('import/output_processed:0')
			
			# Iterate files from directory.
			for f in images:
				start_time = time.time()

				# Read image 
				img = cv.imread(f, 1)
				img_stacked = np.vstack((np.vstack((np.zeros((STACK_SIZE_TOP, 752, 3), dtype=img.dtype), img)), np.zeros((STACK_SIZE_BOTTOM, 752, 3), dtype=img.dtype)))
				
				# Process image that will be evaluated by the model.
				img_pred = imresize(img[:, :, 0], [IM_RESIZE_HEIGHT, IM_RESIZE_WIDTH], 'bilinear')
				img_pred = img_pred.astype(np.float32)
				img_pred = np.multiply(img_pred, 1.0 / 256.0)
				img_pred = img_pred.flatten()

				# Compute prediction point.
				predictions = output_node_processed.eval(
					feed_dict = {
						'import/input_images:0': img_pred,
						'import/keep_prob:0': 1.0
					}
				)

				predictions = np.round(predictions).astype(int)

				# Plot arrowed line showing prediction point.
				cv.arrowedLine(img_stacked, (CENTER_POINT_BOTTOM[0], CENTER_POINT_BOTTOM[1] + STACK_SIZE_TOP), (predictions[0][0], predictions[0][1] + STACK_SIZE_TOP), (0, 0, 255), thickness=2, tipLength=0.05)
				cv.putText(img_stacked, 'Predicted Point Processed: (' + str(int(round(predictions[0][0]))) + ', ' + str(int(round(predictions[0][1]))) + ')', (10, 465 + STACK_SIZE_TOP + STACK_SIZE_BOTTOM), 0, 0.5, (0, 0, 255), thickness=1)
				
				# If annotation data exist, print info and plot arrowed line to annotated point.
				if label_file_exist:
					x = labels.ix[int(f[len(READ_PATH):-4])]['VP_x']
					y = labels.ix[int(f[len(READ_PATH):-4])]['VP_y']
					cv.arrowedLine(img_stacked, (CENTER_POINT_BOTTOM[0], CENTER_POINT_BOTTOM[1] + STACK_SIZE_TOP), (x, y + STACK_SIZE_TOP), (0, 255, 0), thickness=2, tipLength=0.05)
					cv.putText(img_stacked, 'Annotated Point: (' + str(int(round(x))) + ', ' + str(int(round(y))) + ')', (10, 445 + STACK_SIZE_TOP + STACK_SIZE_BOTTOM), 0, 0.5, (0, 255, 0), thickness=1)
					
					v1 = np.subtract(predictions[0], CENTER_POINT_BOTTOM)
					v2 = np.subtract([x, y], CENTER_POINT_BOTTOM)
					angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * (180.0 / np.pi)
					cv.putText(img_stacked, 'Angle Difference (degree): ' + str(angle)[:5], (376, 465 + STACK_SIZE_TOP + STACK_SIZE_BOTTOM), 0, 0.5, (0, 255, 255), thickness=1)
				
				# Compute elapsed time and print iterations/second.
				elapsed_time = time.time() - start_time
				cv.putText(img_stacked, 'Iterations/Second: ' + str(1.0 / elapsed_time)[:5], (376, 445 + STACK_SIZE_TOP + STACK_SIZE_BOTTOM), 0, 0.5, (255, 0, 255), thickness=1)

				# Show image.
				cv.imshow('display', img_stacked)

				# Autoplay or press key to forward.
				if AUTOPLAY:
					cv.waitKey(1)
				else:
					cv.waitKey(0)
				
			
			cv.destroyAllWindows()
		