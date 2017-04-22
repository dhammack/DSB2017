from scipy import ndimage
import numpy as np
import os
import pdb
import cPickle
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys
import scipy
from sklearn.cluster import DBSCAN

print 'loading candidates'

def get_target(file):
	return float(file.split('_')[1].replace('.npy',''))
	
def aggregate_predictions_for_patient(preds):
	#todo
	return
	
def random_perturb(Xbatch,rotate=False): 
	#apply some random transformations...
	swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
	Xcpy = Xbatch.copy()
	for i in range(Xbatch.shape[0]):
		#(1,64,64,64)
		#random 
		Xcpy[i] = Xbatch[i,:,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
		txpose = np.random.permutation([1,2,3])
		Xcpy[i] = np.transpose(Xcpy[i], tuple([0] + list(txpose)))
		if rotate:
			#arbitrary rotation is composition of two
			Xcpy[i,0] = ndimage.interpolation.rotate(Xcpy[i,0], np.random.uniform(-10, 10), axes=(1,0), order=1,reshape=False,cval=-1000,mode='nearest')
			Xcpy[i,0] = ndimage.interpolation.rotate(Xcpy[i,0], np.random.uniform(-10, 10), axes=(2,1), order=1,reshape=False,cval=-1000,mode='nearest')
			
	return Xcpy
	
def get_loc_features(locs, malig_scores, sizes):
	normalized_locs = locs.astype('float32') / sizes.astype('float32')
	
	#location of the most malignant tumor
	loc_from_malig = normalized_locs[ np.argmax(malig_scores) ]
	
	dist_mat = np.zeros((locs.shape[0], locs.shape[0]))
	for i,loc_a in enumerate(locs):
		for j,loc_b in enumerate(locs):
			dist_mat[i,j] = np.mean(np.abs(loc_a - loc_b))
			
	dbs = DBSCAN(eps=60, min_samples=2, metric='precomputed', leaf_size=2).fit(dist_mat)
	num_clusters = np.max(dbs.labels_) + 1
	num_noise = (dbs.labels_ == -1).sum()
	
	#new feature: sum of malig_scores but normalizing by cluster.
	cluster_avgs = []
	for clusternum in range(num_clusters):
		cluster_avgs.append( malig_scores[dbs.labels_ == clusternum].mean())
		
	#now get the -1's
	for i,(clusterix,malig) in enumerate(zip(dbs.labels_,malig_scores)):
		if clusterix == -1:
			cluster_avgs.append(malig)
			
	weighted_sum_malig = np.sum(cluster_avgs)
	weighted_mean_malig = np.mean(cluster_avgs)
	
	#size of biggest cluster
	sizes = np.bincount(dbs.labels_[dbs.labels_ > 0])
	if len(sizes) > 0:
		maxsize = np.max(sizes)
	else:
		maxsize = 1
	n_nodules = float(locs.shape[0])
	
	return np.concatenate([loc_from_malig, normalized_locs.std(axis=0), [float(num_clusters) / n_nodules, float(num_noise) / n_nodules, weighted_mean_malig, float(maxsize) / n_nodules]])
	
def process_voxels(voxels, locs, sizes):
	#mapping from the set of voxels for a patient to a set of features
	
	n_TTA = 3
	diams = []
	lobs = []
	spics = []
	maligs = []
	
	for tta_ix in range(n_TTA):
		#generate normalized predictions for the multi output models...
		Yhats = []
		#NOTE THAT THE FT MODEL ADDED TWICE
		#this is because it was trained on the val set so we didn't use it in the weighted average
		#we used the non ft. however at test time we want to use the ft one.
		for model in [model34, model_multi_relu,model34_repl,model35_relu,model_v36_mse_ft,model_v36_mse_ft,model_v29]:
			Yhats.append(model.predict(random_perturb(voxels), batch_size=32))
		#normalize the ones that require it
		norm_ixs = [1, 3, 4, 5]
		for ix in norm_ixs:
			p = Yhats[ix]
			# assert p[1].max() > 1 and p[2].max() > 1 and p[3].max() > 1
			p[1] /= 5.0; p[2] /= 5.0; p[3] /=5.0
			Yhats[ix] = p
		p = Yhats[6]
		assert len(p) > 4 #and p[1].max() <= 1 and p[2].max() <= 1 and p[3].max() <= 1
		Yhats[6] = p[:4]
		#score the single output (malignancy only) models
		
		Yhats_single = []
		for model in [model_sigmoid, model_relu_s2]:
			Yhats_single.append(model.predict(random_perturb(voxels),batch_size=32))
		#normalize
		norm_ixs = [1]
		for ix in norm_ixs:
			p = Yhats_single[ix]
			# assert p.max() > 1
			p /= 5.0
			Yhats_single[ix] = p
		
		#ok. we've got all the predictions we need here. Let's compute our regressions for each output
		#TODO: replace v36 with the fine tuned v36
		xdiam = np.concatenate([Yhats[0][0], Yhats[1][0], Yhats[2][0],Yhats[3][0],
						   Yhats[5][0],Yhats[6][0]],axis=1)
		xlob = np.concatenate([Yhats[0][1], Yhats[1][1], Yhats[2][1],Yhats[3][1],
							   Yhats[5][1],Yhats[6][1]],axis=1)
		xspic = np.concatenate([Yhats[0][2], Yhats[1][2], Yhats[2][2],Yhats[3][2],
						   Yhats[5][2],Yhats[6][2]],axis=1)
		xmal = np.concatenate([Yhats[0][3], Yhats[1][3], Yhats[2][3],Yhats[3][3], Yhats[5][3],Yhats[6][3],Yhats_single[0],Yhats_single[1]],axis=1)
						   
		
		coefs_diam = [ 0.49502715,  0.04357547,  0.13213973,  0.		,  0.13237931,	   0.23392186]
		coefs_lob = [ 0.		,  0.74029565,  0.01063934,  0.18729332,  0.02203343, 0.13645789]
		coefs_spic = [ 0.33892995,  0.27719817,  0.01419351,  0.21342018,  0.16956094, 0.05932272]
		coefs_malig = [ 0.214,  0.0968,  0.02,  0.16 ,  0.0715,0.0023,  0.359,  0.130]
		
		pred_diam = np.sum(c*xdiam[:,i] for i,c in enumerate(coefs_diam))
		pred_lob = np.sum(c*xlob[:,i] for i,c in enumerate(coefs_lob))
		pred_spic = np.sum(c*xspic[:,i] for i,c in enumerate(coefs_spic))
		pred_malig = np.sum(c*xmal[:,i] for i,c in enumerate(coefs_malig))
		diams.append(pred_diam)
		lobs.append(pred_lob)
		spics.append(pred_spic)
		maligs.append(pred_malig)
	
	#mean taken over ttas
	preds = np.stack([np.mean(diams,axis=0).ravel(), np.mean(lobs,axis=0).ravel(), np.mean(spics,axis=0).ravel(), np.mean(maligs,axis=0).ravel()],axis=1)

	xmax = np.max(preds, axis=0) #taken over voxels.
	xsd = np.std(preds, axis=0)
	
	location_feats = get_loc_features(locs, preds[:,3], sizes)
	return np.concatenate([xmax, xsd, location_feats],axis=0)

# voxels = np.load(sys.argv[1])
# patients = np.load(sys.argv[1].replace('candidate_nodules', 'nodule_patient_map'))
# locations = np.load(sys.argv[1].replace('candidate_nodules', 'nodule_locations'))
# sizes = np.load(sys.argv[1].replace('candidate_nodules', 'img_shapes'))

# assert voxels.shape[0] == patients.shape[0]

# print 'loaded', voxels.shape[0], 'candidates'

model34 = load_model(r"D:\Dlung\model_LUNA_64_v34_describer_24.h5")
model_multi_relu = load_model(r"F:\Flung\descriptor models\model_des_v35_multi_64_24.h5")
model34_repl = load_model(r"F:\Flung\descriptor models\model_des_v34_repl_64_24.h5")
model35_relu = load_model(r"F:\Flung\descriptor models\model_des_v35_relu_64_24_multiout.h5")
model_v36_mse_ft = load_model(r"F:\Flung\descriptor models\model_des_v36_mse_64_finetune_02.h5")
# model_v36_mse = load_model(r"F:\Flung\descriptor models\model_des_v36_mse_64_24.h5")
model_v29 = load_model(r"F:\Flung\model_LUNA_64_v29_14.h5")

#model_relu = load_model(r"F:\Flung\descriptor models\malig_only_v35_relu_b32.")
model_sigmoid = load_model(r"F:\Flung\descriptor models\model_des_v35_sigmoid_64_24.h5")
model_relu_s2 = load_model(r"F:\Flung\descriptor models\model_des_v35_relu_s2_64_24.h5")

all_files = [f for f in os.listdir(r'E:\lung\data_raw\data')]
all_features = []
# all_targets = []
base_path = r'F:\Flung\nodule models\v29_nodules'
import pandas as pd
for i,patient in enumerate(all_files):
	# print f
	
	patient_vox = np.load(os.path.join(base_path, 'vox_' + patient))#voxels[filter]
	patient_locs = np.load(os.path.join(base_path, 'cents_' + patient))
	patient_sizes = np.load(os.path.join(base_path, 'shapes_' + patient))
	
	print patient_vox.shape[0], 'nodules for patient', patient, 'number',i
	features = process_voxels(patient_vox, patient_locs, patient_sizes)
	# target = get_target(patient)
	
	all_features.append(features)
	# all_targets.append(target)
	X = np.stack(all_features)
	
	
np.save('temp.npy', X)
# _,modelname = os.path.split(sys.argv[2])
modelname = 'weighted_ensemble_v1_nodulesv29'
df = pd.DataFrame(data=X, index=all_files)
df.index.name = 'patient'
df.to_csv(modelname + '.csv')

	