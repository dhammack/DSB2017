from scipy import ndimage
import numpy as np
import os
import pdb
import cv2
#TODO: replace with segmentation

def find_start(arr, thresh=.5):
	#determine when the arr first exceeds thresh
	#arr = arr.ravel()
	for i in range(arr.shape[0]):
		if arr[i] > thresh:
			#print 'returning', i
			return np.clip(i - 8, 0, arr.shape[0])
	return 0
	
		
# def crop(img,thresh=-800):
	# sum0 = img.mean(axis=(1,2))
	# sum1 = img.mean(axis=(0,2))
	# sum2 = img.mean(axis=(0,1))
	
	# start0 = find_start(sum0,thresh)+1
	# end0 = -1 *( find_start(sum0[::-1],thresh) + 1)
	# start1 = find_start(sum1,thresh)+1
	# end1 = -1 * (find_start(sum1[::-1],thresh) + 1)
	# start2 = find_start(sum2,thresh)+1
	# end2 = -1* (find_start(sum2[::-1],thresh) +1)
	# assert start0 < int(img.shape[0]*.5) < img.shape[0] - end0, 'bad crop ' + str(start0) + ' ' + str(end0) + ' ' + str(img.shape[0])
	# assert start1 < int(img.shape[1]*.5) < img.shape[1] - end1, 'bad crop ' + str(start1) + ' ' + str(end1) + ' ' + str(img.shape[1])
	# assert start2 < int(img.shape[2]*.5) < img.shape[2] - end2, 'bad crop ' + str(start2) + ' ' + str(end2) + ' ' + str(img.shape[2])
	# assert end0 < 0 and end1 < 0 and end2 < 0, 'one end >= 0'
	# assert start0 > 0 and start1 > 0 and start2 > 0, 'one start <= 0'
	
	# return img[start0:end0,start1:end1,start2:end2]
	
	
def crop_img(img_cpy):
	# img_raw = np.load(patient)
	# downsample = 1
	masks = []
	img_raw = img_cpy.copy()
	
	for i in range(img_raw.shape[2]):
		img_slice = img_raw[ :,:,i]
		img = img_slice.copy()

		img[img>-300] = 255
		img[img<-300] = 0
		img = np.uint8(img)
		contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) > 0:
		

			largest_contour = max(contours, key=cv2.contourArea)
		else:
			mask = (np.zeros(img.shape, np.uint8) < 255)
			masks.append(mask)
			continue
		mask = np.zeros(img.shape, np.uint8)
		cv2.fillPoly(mask, [largest_contour], 255)

#		 imshow(mask); show()

		# apply mask to threshold image to remove outside. this is our new mask
		img = ~img 

		img[(mask == 0)] = 0 # <-- Larger than threshold value


		# apply closing to the mask
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # <- to remove speckles...
		img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

		#the image has an outside part which we don't care about (value 0)
		#and a boundary that we don't care about (value 255)
		#and some noise that we don't care about (value 125)
		mask = (img < 255)
#		 img_raw[~mask] = -2000
#		 imshow(img_raw); colorbar(); show()
		masks.append(mask)
	
	#now we have one mask per slice. To determine our bounding box, take the max x,y,z plus a fuzz factor
	ixs_to_remove = [i for i,m in enumerate(masks) if np.mean(m) > .995]
	
	# masks =[m for m in masks if np.mean(m) < .995]
	masks = np.stack(masks, axis=2)
	masks = np.delete(masks, ixs_to_remove, axis=2)
	img_raw = np.delete(img_raw, ixs_to_remove, axis=2)
	
	
	#0 = mask, 1 = background
	x_dim = np.min(masks, axis=(1,2))
	y_dim = np.min(masks, axis=(0,2))
	z_dim = np.min(masks, axis=(0,1))
	
	xstart = find_start(1 - x_dim, .5)
	xend = -(find_start(1 - x_dim[::-1], .5) + 1)
	
	ystart = find_start(1 - y_dim, .5)
	yend = -(find_start(1 - y_dim[::-1], .5) + 1)
	
	
	zstart = find_start(1 - z_dim, .5)
	zend = -( find_start(1 - z_dim[::-1], .5) + 1)
	
	# try:
	assert xstart < int(img_raw.shape[0]*.5) < img_raw.shape[0] - xend, 'bad crop ' + str(xstart) + ' ' + str(xend) + ' ' + str(img_raw.shape[0])
	assert ystart < int(img_raw.shape[1]*.5) < img_raw.shape[1] - yend, 'bad crop ' + str(ystart) + ' ' + str(yend) + ' ' + str(img_raw.shape[1])
	assert zstart < int(img_raw.shape[2]*.5) < img_raw.shape[2] - zend, 'bad crop ' + str(zstart) + ' ' + str(zend) + ' ' + str(img_raw.shape[2])
	assert xend < 0 and yend < 0 and zend < 0, 'one end >= 0'
	assert xstart >= 0 and ystart >= 0 and zstart >= 0, 'one start <= 0'
	# except AssertionError as e:
		# print 'WARNING cropping failed. using full img', e
		# return img_raw
		
	return img_raw[xstart:xend,ystart:yend,zstart:zend], masks[xstart:xend,ystart:yend,zstart:zend]

def get_strides(steps,size,offset,VOXEL_SIZE):
	if steps * VOXEL_SIZE < size - 2*offset:
		#not enough coverage. start and end are modified
		start = (size - steps*VOXEL_SIZE) / 2
		end = size - start - VOXEL_SIZE
	else:
		start = offset
		end = size-VOXEL_SIZE - offset
	return list(np.around(np.linspace(start,end,steps)).astype('int32'))

def img_to_vox(img,VOXEL_SIZE,mask):

	#mask == 0 -> inside lung.
	#mask == 1 -> outside lung 
	
	#first let's just get the minimum amount of coverage
	samples0 = int(img.shape[0] / float(VOXEL_SIZE)) + 4
	samples1 = int(img.shape[1] / float(VOXEL_SIZE)) + 4
	samples2 = int(img.shape[2] / float(VOXEL_SIZE)) + 4
	
	ixs0 = get_strides(samples0,img.shape[0],0,VOXEL_SIZE)
	ixs1 = get_strides(samples1,img.shape[1],0,VOXEL_SIZE)
	ixs2 = get_strides(samples2,img.shape[2],0,VOXEL_SIZE)

	subvoxels = []
	locations = []
	centroids = []
	for i0,x0 in enumerate(ixs0):
		for i1,x1 in enumerate(ixs1):
			for i2,x2 in enumerate(ixs2):
				if mask[x0:x0+VOXEL_SIZE,x1:x1+VOXEL_SIZE,x2:x2+VOXEL_SIZE].mean() > .99:
					#basically no lung in this voxel, might as well ignore.
					continue
				subvoxels.append(img[x0:x0+VOXEL_SIZE,x1:x1+VOXEL_SIZE,x2:x2+VOXEL_SIZE])
				assert subvoxels[-1].shape == (VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE), 'bad subvoxel shape ' + str(subvoxels[-1].shape) + ' ' + str([x0,x1,x2]) + ' ' + str(img.shape)
				locations.append((i0,i1,i2))
				centroids.append((x0+VOXEL_SIZE/2,x1+VOXEL_SIZE/2,x2+VOXEL_SIZE/2))
	X = np.stack(subvoxels, axis=0)
	#print 'num subvoxels:', X.shape[0]
	X = np.expand_dims(X, 1)
	#normalized locations
	#allows us to de-weight certain places...
	
	return X,locations,centroids

def random_perturb(Xbatch): 
	#apply some random transformations...
	Xcpy = Xbatch.copy()
	swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
	for i in range(Xbatch.shape[0]):
		#(1,32,32,32)
		#random 
		Xcpy[i] = Xbatch[i,:,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
		txpose = np.random.permutation([1,2,3])
		Xcpy[i] = np.transpose(Xcpy[i], tuple([0] + list(txpose)))
	return Xcpy
	
def get_interesting_ixs(preds,thresh=1.5,max_ct=50):
	#return the indices of interest
	ixs = []
	# preds_at_ixs = []
	for i in range(preds.shape[0]):
		if preds[i] > thresh:
			ixs.append(i)
			# preds_at_ixs.append(preds[i])
			
	if len(ixs) == 0:
		ixs = [np.argmax(preds)]
		# preds_at_ixs = [preds[np.argmax(preds)]]
		
	if len(ixs) > max_ct:
		ixs = np.argsort(preds)[-max_ct:]
		
	return np.array(ixs)
	
	
def load_and_txform_file(file,model,VOXEL_SIZE,batch_size,n_TTA=32):
	#read, convert to voxels.
	xorig = np.load(os.path.join(DATA_DIR, file))
	x = np.clip(xorig.copy(), -1000, 400)
	x,mask = crop_img(x)

	x = ((x + 1000.) / (400. + 1000.)).astype('float32')
	voxels, locs, centroids = img_to_vox(x,VOXEL_SIZE,mask)
	#predict on voxels, keep top N ROIs
	preds = model.predict(voxels, batch_size=batch_size).ravel()
	
	topNixs = get_interesting_ixs(preds)
	topNvox = voxels[topNixs]
	topNcentroids = np.array(centroids)[topNixs]
	topNpreds = preds[topNixs]
	
	return topNvox, topNcentroids, [x.shape] * topNcentroids.shape[0], topNpreds

	
DATA_DIR = r'/home/ec2-user/data'

if __name__ == '__main__':
	from keras.models import load_model
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import log_loss
	from joblib import Parallel, delayed
	import sys
	import scipy
	model = load_model(r"F:\Flung\nodule models\model_clf_v1_hq_64_finetune_04.h5")
	from os.path import join
	
	train_files = [f for f in os.listdir(DATA_DIR)]
	VOXEL_SIZE = 64
	model_batch_size=32
	
	all_nodules = []
	nodule_map = []
	all_centroids = []
	all_img_shapes = []
	all_nodule_preds = []
	#base_path = r'F:\Flung\nodule models\v1_nodules'
	base_path = r'F:\Flung\nodule models\stage2_nodules_v1'
	
	for file in train_files:
		
		#return topNvox, topNcentroids, [x.shape] * topNcentroids.shape[0], topNpreds
		vox, cents, shapes, preds = load_and_txform_file(file, model, VOXEL_SIZE, batch_size=32, n_TTA=2)
		np.save(join(base_path, 'vox_' + file), vox)
		np.save(join(base_path, 'cents_' + file), cents)
		np.save(join(base_path, 'shapes_' + file), shapes)
		np.save(join(base_path, 'preds_' + file), preds)
		
		
		
		# all_nodules.extend(list(vox))
		# all_centroids.extend(list(cents))
		# all_img_shapes.extend(list(shapes))
		# all_nodule_preds.extend(list(preds))
		# nodule_map.extend( [file] * len(vox))
		
		# assert len(all_nodules) == len(nodule_map) == len(all_centroids) == len(all_img_shapes), 'mapping error'
	
		print file, vox.shape[0], 'nodules', preds.max(), 'max score'
		
		
	# np.save('X_candidate_nodules_' + sys.argv[1], np.stack(all_nodules))
	# np.save('nodule_patient_map_' + sys.argv[1], np.array(nodule_map))
	# np.save('nodule_locations_' + sys.argv[1], np.array(all_centroids))
	# np.save('img_shapes_' + sys.argv[1], np.array(all_img_shapes))
	# np.save('nodule_scores_' + sys.argv[1], np.array(all_nodule_preds))
	
	
		
	
	
		
	
