from scipy import ndimage
import numpy as np
import os
import pdb

def find_start(arr, thresh=-800):
	#determine when the arr first exceeds thresh
	#arr = arr.ravel()
	for i in range(arr.shape[0]):
		if arr[i] > thresh:
			#print 'returning', i
			return np.clip(i - 5, 0, arr.shape[0])
	return 0
	
		
def crop(img,thresh=-800):
	sum0 = img.mean(axis=(1,2))
	sum1 = img.mean(axis=(0,2))
	sum2 = img.mean(axis=(0,1))
	
	start0 = find_start(sum0,thresh)+1
	end0 = -1 *( find_start(sum0[::-1],thresh) + 1)
	start1 = find_start(sum1,thresh)+1
	end1 = -1 * (find_start(sum1[::-1],thresh) + 1)
	start2 = find_start(sum2,thresh)+1
	end2 = -1* (find_start(sum2[::-1],thresh) +1)
	assert start0 < int(img.shape[0]*.5) < img.shape[0] - end0, 'bad crop ' + str(start0) + ' ' + str(end0) + ' ' + str(img.shape[0])
	assert start1 < int(img.shape[1]*.5) < img.shape[1] - end1, 'bad crop ' + str(start1) + ' ' + str(end1) + ' ' + str(img.shape[1])
	assert start2 < int(img.shape[2]*.5) < img.shape[2] - end2, 'bad crop ' + str(start2) + ' ' + str(end2) + ' ' + str(img.shape[2])
	assert end0 < 0 and end1 < 0 and end2 < 0, 'one end >= 0'
	assert start0 > 0 and start1 > 0 and start2 > 0, 'one start <= 0'
	
	return img[start0:end0,start1:end1,start2:end2]
	

def get_strides(steps,size,offset,VOXEL_SIZE):
	if steps * VOXEL_SIZE < size - 2*offset:
		#not enough coverage. start and end are modified
		start = (size - steps*VOXEL_SIZE) / 2
		end = size - start - VOXEL_SIZE
	else:
		start = offset
		end = size-VOXEL_SIZE - offset
	return list(np.around(np.linspace(start,end,steps)).astype('int32'))

def img_to_vox(img,VOXEL_SIZE):

	# img = np.clip(img, -1000, 400)
	# img = ((img + 1000.) / (400. + 1000.)).astype('float32')
	
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
	swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
	for i in range(Xbatch.shape[0]):
		#(1,64,64,64)
		#random 
		Xbatch[i] = Xbatch[i,:,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
		txpose = np.random.permutation([1,2,3])
		Xbatch[i] = np.transpose(Xbatch[i], tuple([0] + list(txpose)))
	return Xbatch
	
def get_fuzzed_view(img, center, n_view,VOXEL_SIZE):
	max_fuzz = VOXEL_SIZE/4
	halfsize = VOXEL_SIZE/2
	voxels = []
	for i in range(n_view):
		fuzz = np.random.randint(-max_fuzz, max_fuzz+1,3)
		fuzzed_center = fuzz + center
		fuzzed_center[0] = np.clip(fuzzed_center[0], halfsize, img.shape[0] - halfsize)
		fuzzed_center[1] = np.clip(fuzzed_center[1], halfsize, img.shape[1] - halfsize)
		fuzzed_center[2] = np.clip(fuzzed_center[2], halfsize, img.shape[2] - halfsize)
		
		
		# pdb.set_trace()
		voxels.append(img[fuzzed_center[0]-halfsize:fuzzed_center[0]+halfsize,fuzzed_center[1]-halfsize:fuzzed_center[1]+halfsize,fuzzed_center[2]-halfsize:fuzzed_center[2]+halfsize])
		#print voxels[-1].shape, fuzzed_center
	return voxels
	
def get_fuzzed_loc_view(img, center,VOXEL_SIZE):
	fuzz = VOXEL_SIZE/4
	halfsize = VOXEL_SIZE/2
	voxels = []
	sign = [-1,1]
	centers = []
	for s1 in sign:
		for s2 in sign:
			for s3 in sign:
				fuzzed_center = center + np.array([s1*fuzz,s2*fuzz,s3*fuzz])
				fuzzed_center[0] = np.clip(fuzzed_center[0], halfsize, img.shape[0] - halfsize)
				fuzzed_center[1] = np.clip(fuzzed_center[1], halfsize, img.shape[1] - halfsize)
				fuzzed_center[2] = np.clip(fuzzed_center[2], halfsize, img.shape[2] - halfsize)
				voxels.append(img[fuzzed_center[0]-halfsize:fuzzed_center[0]+halfsize,fuzzed_center[1]-halfsize:fuzzed_center[1]+halfsize,fuzzed_center[2]-halfsize:fuzzed_center[2]+halfsize])
				centers.append(fuzzed_center)
	return voxels, centers

def get_location_features(centroid,img_shape):
	#normalize by image size
	#don't worry about left/right or front/back. just top/bottom
	return np.array(centroid).astype('float32') / np.array(img_shape).astype('float32')
	
# def get_32sized_subvoxels(img, center):
	# #get the subvoxels for this part of the image
	# subimg = img[center[0]-32:center[0]+32,center[1]-32:center[1]+32,center[2]-32:center[2]+32]
	# result = img_to_vox(subimg, 32)
	# return result[0], result[2]

def get_subvox_at(img, center,size=64):
	subimg = img[center[0]-size/2:center[0]+size/2,center[1]-size/2:center[1]+size/2,center[2]-size/2:center[2]+size/2]
	return np.expand_dims(subimg, 0) #(1,size,size,size)
	
def get_interesting_ixs(preds):
	#return the indices of interest
	ixs = []
	for i in range(preds.shape[0]):
		if preds[i,0] > 5:
			ixs.append(i)
		elif preds[i,1] > 0.3:
			ixs.append(i)
		elif preds[i,2] > 0.3:
			ixs.append(i)
		elif preds[i,3] > 0.3:
			ixs.append(i)
	
	if len(ixs) == 0:
		ixs = [np.argmax(preds[:,3])]
	return np.array(ixs)
	
	
def load_and_txform_file(file,model,VOXEL_SIZE,batch_size):
	#read, convert to voxels.
	xorig = np.load(os.path.join(DATA_DIR, file))
	x = np.clip(xorig, -1000, 400)
	try:
		x = crop(x)
	except:
		print 'couldnt crop', file
		print 'trying again with diff threshold'
		try:
			x = crop(x,-900)
		except:
			print 'still couldnt crop.'
			try:
				x = crop(x, -1000)
			except:
				print 'failed to crop at all :('
				exit()

	x = ((x + 1000.) / (400. + 1000.)).astype('float32')
	voxels, locs, centroids = img_to_vox(x,VOXEL_SIZE)
	#predict on voxels, keep top N ROIs
	# pdb.set_trace()
	preds = model.predict(voxels, batch_size=batch_size)

	if type(preds) == list:
		preds = np.concatenate(preds,axis=1)
	
	topNixs = get_interesting_ixs(preds)
	
	topNvox = voxels[topNixs]
	topNcentroids = np.array(centroids)[topNixs]
	

	return topNvox, topNcentroids, [x.shape] * topNcentroids.shape[0]

	
	
# def rough_malig_score(preds,diam_ix=0,spic_ix=18,malig_ix=19):
	# return scipy.special.expit(-9.02 + preds[:,diam_ix] * .055 + preds[:,spic_ix] * -1.6 + preds[:,malig_ix] * 10.5)
# 0 0.000822845018192
# 1 1.30373429162
# 2 10.4641008485
# [-9.56203379]
def rough_malig_score(preds,diam_ix=0,lob_ix=1,malig_ix=3):
	return scipy.special.expit(-9.5 + preds[:,diam_ix] * .0008228 + preds[:,lob_ix] * 1.304 + preds[:,malig_ix] * 10.45)
	
def aggregate_tta_preds(preds):
	#this used to be a straight maximum but I think we can do better. 
	result = np.zeros(preds.shape[1])
	rough_malig_scores = rough_malig_score(preds)
	return preds[np.argmax(rough_malig_scores),:]
	
	# result[0] = preds[:,0].max() # max diameter
	# result[1] = preds[:,1].min() #0 = nodule, 1 = no nodule.
	# result[[2,3,4]] = preds[:,[2,3,4]].max() #categories of texture. 
	
DATA_DIR = r'F:\Flung\stage2\1mm'

if __name__ == '__main__':
	from keras.models import load_model
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import log_loss
	from joblib import Parallel, delayed
	import sys
	import scipy
	model = load_model(r"D:\Dlung\model_LUNA_64_v29_14.h5")
	from os.path import join
	
	train_files = [f for f in os.listdir(DATA_DIR)]
	#score the model as we go
	
	VOXEL_SIZE = 64
	model_batch_size=64
	base_path = r'F:\Flung\stage2\v29_nodules'
	
	for i,file in enumerate(train_files):
		vox,cents,shapes = load_and_txform_file(file, model, VOXEL_SIZE, model_batch_size)
		print file, i, 'of', len(train_files),'n_nod',vox.shape[0]
		np.save(join(base_path, 'vox_' + file), vox)
		np.save(join(base_path, 'cents_' + file), cents)
		np.save(join(base_path, 'shapes_' + file), np.array(shapes))
		
	
		
	