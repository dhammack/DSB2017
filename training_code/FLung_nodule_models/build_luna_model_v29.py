
def conv_block(x_input, num_filters,pool=True,norm=False,drop_rate=0.0):
	
	x1 = Convolution3D(num_filters,3,3,3,border_mode='same',W_regularizer=l2(1e-4))(x_input)
	if norm:
		x1 = BatchNormalization(axis=1)(x1)
		#x1 = Lambda(relu_norm)(x1)
	if drop_rate > 0.0:
		x1 = GaussianDropout(drop_rate)(x1)
		
	x1 = LeakyReLU(.1)(x1)
	if pool:
		x1 = MaxPooling3D()(x1)
	x_out = x1
	return x_out
	
	
def str_to_arr(arr_str,length):
	while '  ' in arr_str:
		arr_str = arr_str.replace('  ', ' ')
	result = eval(arr_str.replace('[ ', '[').replace(' ]', ']').replace(' ', ','))
	assert len(result) == length
	return np.array(result)
	
def dense_branch(xstart, name, outsize=1,activation='sigmoid'):
	xdense_ = Dense(32,W_regularizer=l2(1e-4))(xstart)
	xdense_ = BatchNormalization()(xdense_)
	# xdense_ = GaussianDropout(0)(xdense_)
	xdense_ = LeakyReLU(.1)(xdense_)
	xout = Dense(outsize,activation=activation, name=name,W_regularizer=l2(1e-4))(xdense_)
	return xout
	
def build_model(input_shape):

	xin = Input(input_shape)
	
	#shift the below down by one
	x1 = conv_block(xin,8,norm=True,drop_rate=0) #outputs 9 ch
	x1_ident = AveragePooling3D()(xin)
	x1_merged = merge([x1, x1_ident],mode='concat', concat_axis=1)
	
	x2_1 = conv_block(x1_merged,24,norm=True,drop_rate=0) #outputs 16+9 ch  = 25
	x2_ident = AveragePooling3D()(x1_ident)
	x2_merged = merge([x2_1,x2_ident],mode='concat', concat_axis=1)
	
	#by branching we reduce the #params
	x3_1 = conv_block(x2_merged,64,norm=True,drop_rate=0) #outputs 25 + 16 ch = 41
	x3_ident = AveragePooling3D()(x2_ident)
	x3_merged = merge([x3_1,x3_ident],mode='concat', concat_axis=1)
	
	x4_1 = conv_block(x3_merged,72,norm=True,drop_rate=0) #outputs 25 + 16 ch = 41
	x4_ident = AveragePooling3D()(x3_ident)
	x4_merged = merge([x4_1,x4_ident],mode='concat', concat_axis=1)
	
	
	x5_1 = conv_block(x4_merged,72,norm=True,pool=False,drop_rate=0) #outputs 25 + 16 ch = 41
	
	xpool = GlobalMaxPooling3D()(x5_1)
	xpool_norm = BatchNormalization()(xpool)
	#xpool_norm = GaussianDropout(.1)(xpool_norm)
	
	#from here let's branch and predict different things
	xout_diam= dense_branch(xpool_norm,name='o_d',outsize=1,activation='relu')
	
	#sphericity
	# xout_spher= dense_branch(xpool_norm,name='o_spher',outsize=4,activation='softmax')
	
	# xout_text = dense_branch(xpool_norm,name='o_t',outsize=4,activation='softmax')
	
	#calcification
	# xout_calc = dense_branch(xpool_norm,name='o_c',outsize=7,activation='softmax')
	xout_cad_falsepositive = dense_branch(xpool_norm, name='o_fp',outsize=3,activation='softmax')
	
	# xout_cat = merge([xout_text,xout_spher,xout_calc],name='o_cat',mode='concat', concat_axis=1)
	
	
	xout_margin = dense_branch(xpool_norm,name='o_marg',outsize=1,activation='sigmoid')
	xout_lob = dense_branch(xpool_norm,name='o_lob',outsize=1,activation='sigmoid')
	xout_spic = dense_branch(xpool_norm,name='o_spic',outsize=1,activation='sigmoid')
	xout_malig = dense_branch(xpool_norm,name='o_mal',outsize=1,activation='sigmoid')
	
	# xout_numeric = merge([xout_margin, xout_lob, xout_spic, xout_malig],name='o_num',mode='concat',concat_axis=1)
	
	
	
	
	model = Model(input=xin,output=[xout_diam, xout_lob, xout_spic, xout_malig, xout_cad_falsepositive])
	
	if input_shape[1] == 32:
		lr_start = .005
	elif input_shape[1] == 64:
		lr_start = .001
	elif input_shape[1] == 128:
		lr_start = 1e-4
	elif input_shape[1] == 96:
		lr_start = 5e-4
	
		
	opt = Nadam(lr_start,clipvalue=1.0)
	print 'compiling model'

	model.compile(optimizer=opt,loss={'o_d':'mse', 'o_lob':'binary_crossentropy', 'o_spic':'binary_crossentropy', 
										'o_mal':'binary_crossentropy', 'o_fp':'categorical_crossentropy'},
								loss_weights={'o_d':1.0, 'o_lob':5.0, 'o_spic':5.0, 'o_mal':5.0, 'o_fp':5.0})
	return model
	
def random_perturb(Xbatch): 
	#apply some random transformations...
	swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
	for i in range(Xbatch.shape[0]):
		#(1,32,32,32)
		#random 
		Xbatch[i] = Xbatch[i,:,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
		txpose = np.random.permutation([1,2,3])
		Xbatch[i] = np.transpose(Xbatch[i], tuple([0] + list(txpose)))
	return Xbatch
	
def log_abs_error(y_true, y_pred):
	return -1 * K.mean(K.log(K.clip(1.0 - K.abs(y_true - y_pred), K.epsilon(), None)))
	
def get_generator_static(X1,X2,IX1,IX2,batch_size,augment=True,current_frac=.75):
	
	df = pd.read_csv(r"D:\Dlung\annotations_enhanced.csv")
	Ydiam = df['diameter_mm'].values.astype('float32')
	Ycalc = np.zeros((df.shape[0],7)).astype('float32')
	Yspher = np.zeros((df.shape[0],4)).astype('float32')
	Ytext = np.zeros((df.shape[0],4)).astype('float32')
		
	df['calcification'] = df['calcification'].apply(lambda x: str_to_arr(x,6))
	df['sphericity'] = df['sphericity'].apply(lambda x: str_to_arr(x,3))
	df['texture'] = df['texture'].apply(lambda x: str_to_arr(x,3))
	
	Ycad_fp = np.zeros((df.shape[0],3)).astype('float32')
	Ycad_fp[:,0] = 1.0
	#3 categories - [nodule, cad false positive, random subvoxel]
	
	
	Ycalc[:,1] = df['calcification'].apply(lambda x: x[0]).values
	Ycalc[:,2] = df['calcification'].apply(lambda x: x[1]).values
	Ycalc[:,3] = df['calcification'].apply(lambda x: x[2]).values
	Ycalc[:,4] = df['calcification'].apply(lambda x: x[3]).values
	Ycalc[:,5] = df['calcification'].apply(lambda x: x[4]).values
	Ycalc[:,6] = df['calcification'].apply(lambda x: x[5]).values

	Yspher[:,1] = df['sphericity'].apply(lambda x: x[0]).values
	Yspher[:,2] = df['sphericity'].apply(lambda x: x[1]).values
	Yspher[:,3] = df['sphericity'].apply(lambda x: x[2]).values

	Ytext[:,1] = df['texture'].apply(lambda x: x[0]).values
	Ytext[:,2] = df['texture'].apply(lambda x: x[1]).values
	Ytext[:,3] = df['texture'].apply(lambda x: x[2]).values
	
	df[['margin', 'lobulation', 'spiculation', 'malignancy']] /= 5.0
	
	Ymargin = df['margin'].values
	Ylob = df['lobulation'].values
	Yspic = df['spiculation'].values
	Ymalig = df['malignancy'].values
	
	# print 'baseline_mse', current_frac * Y1.var() + (1 - current_frac) * Y2.var()
	while True:
		
		n1 = int(current_frac * batch_size)
		n2 = batch_size - n1
		
		ixs1 = np.random.choice(range(X1.shape[0]),size=n1,replace=False)
		ixs2 = np.random.choice(range(X2.shape[0]),size=n2,replace=False)
		Xbatch1,IXbatch1 = X1[ixs1],IX1[ixs1]
		Xbatch2,IXbatch2 = X2[ixs2],IX2[ixs2]
		
		Xbatch = np.concatenate([Xbatch1, Xbatch2],axis=0)
		IXbatch = np.concatenate([IXbatch1, IXbatch2], axis=0)
		IXbatch_eq_neg2 = (IXbatch == -2)
		IXbatch[IXbatch_eq_neg2] = -1
		#this way things that are -2 (cad false positives)
		#get labeled as non-nodules for everything else.
		#but we still know where they are.
		IXbatch_eq_neg1 = (IXbatch == -1)
		
		#making sure we don't use these.
		#row 32 is a big tumor so it should be obvious
		#IXbatch[IXbatch_eq_neg1] = 32
		#IXbatch[IXbatch_eq_neg2] = 32
		
		#normalize
		Xbatch = np.expand_dims(Xbatch, 1)
		if augment:
			Xbatch = random_perturb(Xbatch)
		Xbatch = Xbatch.astype('float32')
		Xbatch = (Xbatch +1000.) / (400. + 1000.)
		Xbatch = np.clip(Xbatch,0,1)
		
		Ybatch_diam = Ydiam[IXbatch]
		Ybatch_diam[IXbatch_eq_neg1] = 0.0
		
		# Ybatch_text = Ytext[IXbatch]
		# Ybatch_text[IXbatch_eq_neg1,1:] = 0.0
		# Ybatch_text[IXbatch_eq_neg1,0] = 1.0
		
		# Ybatch_spher = Yspher[IXbatch]
		# Ybatch_spher[IXbatch_eq_neg1,1:] = 0.0
		# Ybatch_spher[IXbatch_eq_neg1,0] = 1.0
		
		# Ybatch_calc = Ycalc[IXbatch]
		# Ybatch_calc[IXbatch_eq_neg1,1:] = 0.0
		# Ybatch_calc[IXbatch_eq_neg1,0] = 1.0

		# Ybatch_margin = Ymargin[IXbatch]
		# Ybatch_margin[IXbatch_eq_neg1] = 0.0
		
		Ybatch_lob = Ylob[IXbatch]
		Ybatch_lob[IXbatch_eq_neg1] = 0.0
		
		Ybatch_spic = Yspic[IXbatch]
		Ybatch_spic[IXbatch_eq_neg1] = 0.0
		
		Ybatch_malig = Ymalig[IXbatch]
		Ybatch_malig[IXbatch_eq_neg1] = 0.0
		
		Ybatch_cad_fp = Ycad_fp[IXbatch]
		#class 0 things = nodules are already set.
		
		#-1 = random subvoxel, class 3
		Ybatch_cad_fp[IXbatch_eq_neg1,2] = 1.0
		Ybatch_cad_fp[IXbatch_eq_neg1,0] = 0.0
		Ybatch_cad_fp[IXbatch_eq_neg1,1] = 0.0
		
		#-2 = cad false positive, class 2
		Ybatch_cad_fp[IXbatch_eq_neg2,1] = 1.0
		Ybatch_cad_fp[IXbatch_eq_neg2,0] = 0.0
		Ybatch_cad_fp[IXbatch_eq_neg2,2] = 0.0
		
		
		
		
		# Ybatch_categorical = np.concatenate([Ybatch_text,Ybatch_spher,Ybatch_calc],axis=1)
		# Ybatch_numeric = np.concatenate([Ybatch_margin.reshape((-1,1)),Ybatch_lob.reshape((-1,1)),Ybatch_spic.reshape((-1,1)),Ybatch_malig.reshape((-1,1))],axis=1)
		
		
		yield Xbatch, {'o_d':Ybatch_diam, 'o_lob':Ybatch_lob, 'o_spic':Ybatch_spic, 'o_mal':Ybatch_malig, 'o_fp':Ybatch_cad_fp}
	

def stage_1_lr_schedule(i):
	if i < 10:
		return np.float32(.004)
	if i < 15:
		return np.float32(.002)
	if i < 20:
		return np.float32(.0005)
	if i < 23:
		return np.float32(.0001)
	return np.float32(3e-5)
	
def stage_2_lr_schedule(i):
	if i == 0:
		return np.float32(.001)
	if i < 5:
		return np.float32(.0004)
	if i < 10:
		return np.float32(.0002)
	return np.float32(3e-5)
	
def get_batch_size_for_stage(stage):
	
	#return np.around( 64. * (64 ** 3) / (stage ** 3))
	if stage == 32:
		batch_size=128
	if stage == 64:
		batch_size=64
	if stage == 128:
		batch_size=8
	if stage == 256:
		batch_size=2
	if stage == 72:
		batch_size=40
	if stage == 65:
		batch_size=63
	if stage == 96:
		batch_size=18
	return batch_size
	
	
def train_model_on_stage(stage,model,fast_start=False):
	test_mode=False
	split=False
	import time
	
	batch_size=get_batch_size_for_stage(stage)
	if not fast_start:
		#START THE DATA GENERATION PROCESS
		data_gen_process = Process(target=data_generator_fn.main,args=[stage,np.random.randint(0,10000)])
		data_gen_process.start()
		
		print 'waiting for data generator process to finish first chunk'
		while data_gen_process.is_alive():
			time.sleep(1)
	print 'starting training'
	
	# assert split == True, 'TODO: write code for no split'
	
	
	if split:
		Xpos = np.load(r"D:\Dlung\Xpositive_temp_v5.npy")
		IXpos = np.load(r"D:\Dlung\Ixpositive_temp_v5.npy")
		split1 = int(.75*Xpos.shape[0])
		Xtrain1 = Xpos[:split1]
		Xvalid1 = Xpos[split1:]
		IXtrain1 = IXpos[:split1]
		IXvalid1 = IXpos[split1:]
		del Xpos, IXpos
		
		
		Xneg = np.load(r"D:\Dlung\Xrandom_temp_v5.npy")
		IXneg = np.load(r"D:\Dlung\IXrandom_temp_v5.npy")
		#if split:
		split2 = int(.75*Xneg.shape[0])
		Xtrain2 = Xneg[:split2]
		Xvalid2 = Xneg[split2:]
		IXtrain2 = IXneg[:split2]
		IXvalid2 = IXneg[split2:]
		del Xneg, IXneg
		
		train_generator_75 = get_generator_static(Xtrain1,Xtrain2,IXtrain1,IXtrain2, augment=True, batch_size=batch_size,current_frac=.33)
		valid_generator_75 = get_generator_static(Xvalid1,Xvalid2,IXvalid1,IXvalid2, augment=True, batch_size=batch_size,current_frac=.33)
	else:
		Xpos = np.load(r"D:\Dlung\Xpositive_temp_v5.npy")
		IXpos = np.load(r"D:\Dlung\Ixpositive_temp_v5.npy")
		Xneg = np.load(r"D:\Dlung\Xrandom_temp_v5.npy")
		IXneg = np.load(r"D:\Dlung\IXrandom_temp_v5.npy")
		train_generator_75 = get_generator_static(Xpos,Xneg,IXpos,IXneg, augment=True, batch_size=batch_size,current_frac=.33)

				
	#combined_generator = combine_generators(train_generator_ez, valid_generator_ez, frac1=.5)
	name = 'model_LUNA_' + str(stage) + '_v29_{epoch:02d}.h5'
	chkp = ModelCheckpoint(filepath=name)
	#lr_reducer = ReduceLROnPlateau(monitor='loss', factor=.5, patience=4,min_lr=1e-5,epsilon=1e-2,verbose=1)
	if stage == 32:
		lr_schedule = LearningRateScheduler(stage_1_lr_schedule)
		nb_epoch = 25
		samples_per_epoch = 150
	else:
		lr_schedule = LearningRateScheduler(stage_2_lr_schedule)
		nb_epoch=15
		samples_per_epoch = 150
		
	print 'restarting data gen process'
	data_gen_process = Process(target=data_generator_fn.main,args=[stage,np.random.randint(0,10000)])
	data_gen_process.start()
	
	for epoch in range(nb_epoch):
		#check after each epoch if the data is done and if so reload it
		if not data_gen_process.is_alive():
			#reload data
			print 'RELOADING DATA'
			if split:
				del Xtrain1, Xtrain2, Xvalid1, Xvalid2
				Xpos = np.load(r"D:\Dlung\Xpositive_temp_v5.npy")
				IXpos = np.load(r"D:\Dlung\Ixpositive_temp_v5.npy")
				
				#if split
				split1 = int(.75*Xpos.shape[0])
				Xtrain1 = Xpos[:split1]
				Xvalid1 = Xpos[split1:]
				IXtrain1 = IXpos[:split1]
				IXvalid1 = IXpos[split1:]
				del Xpos, IXpos
				
				
				Xneg = np.load(r"D:\Dlung\Xrandom_temp_v5.npy")
				IXneg = np.load(r"D:\Dlung\IXrandom_temp_v5.npy")
				#if split:
				split2 = int(.75*Xneg.shape[0])
				Xtrain2 = Xneg[:split2]
				Xvalid2 = Xneg[split2:]
				IXtrain2 = IXneg[:split2]
				IXvalid2 = IXneg[split2:]
				del Xneg, IXneg
				train_generator_75 = get_generator_static(Xtrain1,Xtrain2,IXtrain1,IXtrain2, augment=True, batch_size=batch_size,current_frac=.33)
				valid_generator_75 = get_generator_static(Xvalid1,Xvalid2,IXvalid1,IXvalid2, augment=True, batch_size=batch_size,current_frac=.33)
			else:
				Xpos = np.load(r"D:\Dlung\Xpositive_temp_v5.npy")
				IXpos = np.load(r"D:\Dlung\Ixpositive_temp_v5.npy")
				Xneg = np.load(r"D:\Dlung\Xrandom_temp_v5.npy")
				IXneg = np.load(r"D:\Dlung\IXrandom_temp_v5.npy")
				train_generator_75 = get_generator_static(Xpos,Xneg,IXpos,IXneg, augment=True, batch_size=batch_size,current_frac=.33)
				
			
			
			#restart data generator
			data_gen_process = Process(target=data_generator_fn.main,args=[stage,np.random.randint(0,10000)])
			data_gen_process.start()
		else:
			print 'data generator still running'
			
		print 'epoch', epoch, 'model lr', model.optimizer.lr.get_value()
		if split:
			model.fit_generator(train_generator_75,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[chkp,lr_schedule],
								validation_data=valid_generator_75, nb_val_samples=samples_per_epoch*batch_size/2,initial_epoch=epoch)
		else:
			model.fit_generator(train_generator_75,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[chkp,lr_schedule],
								initial_epoch=epoch)
	
		
	return model
		
if __name__ == '__main__':

	import pandas as pd
	import pdb
	import numpy as np
	from keras.models import Model, load_model
	from keras.layers import Input, Lambda, Dense, Flatten, Reshape, merge, Highway, Activation,Dropout
	from keras.layers.convolutional import Convolution3D
	from keras.layers.pooling import GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D,GlobalAveragePooling3D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.noise import GaussianDropout
	from keras.optimizers import Adamax, Adam, Nadam
	from keras.layers.advanced_activations import ELU,PReLU,LeakyReLU
	from keras import backend as K
	from sklearn.preprocessing import LabelBinarizer
	from keras.regularizers import l2
	from pylab import imshow, show
	import cPickle as pickle
	from keras.preprocessing.image import ImageDataGenerator
	from keras.models import load_model
	import os
	from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
	from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
	from sklearn.linear_model import LogisticRegression
	import numpy as np # linear algebra
	import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
	import pydicom as dicom
	import os
	import scipy.ndimage
	import matplotlib.pyplot as plt
	from joblib import Parallel, delayed
	from skimage import measure, morphology
	import SimpleITK as sitk
	from PIL import Image
	from scipy import ndimage
	import threading
	from multiprocessing import Process
	import data_generator_fn
	from keras.regularizers import l2
	
	np.random.seed(21345)
	print 'loading data'
	
	stages = [32, 64, 128, 256]
	
	#first build the model on 32 sized
	#this is fast.
		
	#using densenet16 gave 4.69 train loss @32, and 4.98 val loss after epoch 35
	#trying multi conv
	
	#using split net gave 4.2-4.3 train loss and 4.3-4.5 val loss (best so far)
	
	#using big full bn net gave 3.55 train loss and 4.27 val loss (best so far)
	
	#doing same thing but with more dropout gives 4.71 train loss and 7.87 val loss
	
	#less dropout and a tiny bit of l2 regularization gives 3.84 train and 4.98 val (with max and avg pool split)
	
	#turning off dropout entirely gives 3.44 train loss and 4.45 valid loss
	
	#i think data augmentation is better than dropout. i'm going to increase the size of each epoch
	#this should allow for more data aug steps and better training
	
	#FIXED BUG.
	#NEW BASELINE
	#64 model: train loss 3.7 @ 14 and val 3.97 @ 14
	
	#v29 64 model: 4.5 train loss and 4.27 valid loss @ 15
	
	#so clearly the v29 model isn't as good as the v24. 
	#the major changes here were removing the branching and replacing it with 1x1 convs.
	
	
	model_32 = build_model((1,32,32,32))
	model_64 = build_model((1,64,64,64))
	# model_96 = build_model((1,96,96,96))
	 # model_256 = build_model((1,256,256,256))
	
	model_32.summary()
	# model_32.load_weights('model_v29_weights_temp.h5',by_name=True)
	
	model_32 = train_model_on_stage(32, model_32,fast_start=False)
	# #now save these weights and reload them in the next model
	model_32.save_weights('model_v29_weights_temp.h5')
	model_64.load_weights('model_v29_weights_temp.h5')
	model_64 = train_model_on_stage(64, model_64)
	model_64.save_weights('model_v29_weights_temp64.h5')
	# model_96.load_weights('model_v29_weights_temp64.h5')
	# model_96 = train_model_on_stage(96, model_96)
	# model_96.save_weights('model_v29_weights_temp96.h5')
	
	# model_128.load_weights('model_v29_weights_temp64.h5')
	# model_128 = train_model_on_stage(128, model_128)
	# model_128.save_weights('model_v29_weights_temp128.h5')
	
	# model_256.load_weights('model_v29_weights_temp128.h5')
	# model_256 = train_model_on_stage(256, model_256)
	# model_256.save_weights('model_v29_weights_temp256.h5')
	
	
	#model_72.load_weights('model_20_weights_temp64.h5')
	#model_72 = load_model(r"D:\Dlung\model_LUNA_72_v21_00.h5")
	#model_72 = train_model_on_stage(72, model_72)
	#model_72.save_weights('model_21_weights_temp72.h5')
		
	
	#ok. now let's use this model to initialize the bigger ones
	
	
	print 'done.'
