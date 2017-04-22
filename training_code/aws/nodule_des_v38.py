
def conv_block(x_input, num_filters,pool=True,activation='relu',init='orthogonal'):
	
	x1 = Convolution3D(num_filters,3,3,3,border_mode='same',W_regularizer=l2(1e-5),init=init)(x_input)
	x1 = BatchNormalization(axis=1,momentum=0.995)(x1)
	x1 = LeakyReLU(.1)(x1)
	x1 = Convolution3D(num_filters / 2,3,3,3, border_mode='same',W_regularizer=l2(1e-4))(x1)
	x1 = BatchNormalization(axis=1)(x1)
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
	xdense_ = Dense(32,W_regularizer=l2(1e-5))(xstart)
	xdense_ = BatchNormalization(momentum=0.995)(xdense_)
	# xdense_ = GaussianDropout(0)(xdense_)
	xdense_ = LeakyReLU(.1)(xdense_)
	xout = Dense(outsize,activation=activation, name=name,W_regularizer=l2(1e-4))(xdense_)
	return xout
	
def build_model(input_shape):

	xin = Input(input_shape)
	
	#shift the below down by one
	x1 = conv_block(xin,12,activation='relu')
	x1_ident = AveragePooling3D()(xin)
	x1_merged = merge([x1, x1_ident],mode='concat', concat_axis=1)
	
	x2_1 = conv_block(x1_merged,36,activation='relu') #outputs 37 ch
	x2_ident = AveragePooling3D()(x1_ident)
	x2_merged = merge([x2_1,x2_ident],mode='concat', concat_axis=1)
	
	#by branching we reduce the #params
	x3_ident = AveragePooling3D()(x2_ident)
	x3_malig = conv_block(x2_merged,64,activation='relu') #outputs 25 + 16 ch = 41
	x3_malig_merged = merge([x3_malig,x3_ident],mode='concat', concat_axis=1)
	
	x4_ident = AveragePooling3D()(x3_ident)
	x4_malig = conv_block(x3_malig_merged,72,activation='relu') #outputs 25 + 16 ch = 41
	x4_merged = merge([x4_malig,x4_ident],mode='concat', concat_axis=1)
	
	
	x5_malig = conv_block(x4_merged,64) #outputs 25 + 16 ch = 41
	xpool_malig = BatchNormalization(momentum=0.995)(GlobalMaxPooling3D()(x5_malig))
	xout_malig = Dense(1, name='o_mal', activation='softplus')(xpool_malig) #relu output

	x5_diam = conv_block(x4_merged,64) #outputs 25 + 16 ch = 41
	xpool_diam = BatchNormalization(momentum=0.995)(GlobalMaxPooling3D()(x5_diam))
	xout_diam = Dense(1, name='o_diam', activation='softplus')(xpool_diam) #relu output

	x5_lob = conv_block(x4_merged,64) #outputs 25 + 16 ch = 41
	xpool_lob = BatchNormalization(momentum=0.995)(GlobalMaxPooling3D()(x5_lob))
	xout_lob = Dense(1, name='o_lob', activation='softplus')(xpool_lob) #relu output

	x5_spic = conv_block(x4_merged,64) #outputs 25 + 16 ch = 41
	xpool_spic = BatchNormalization(momentum=0.995)(GlobalMaxPooling3D()(x5_spic))
	xout_spic = Dense(1, name='o_spic', activation='softplus')(xpool_spic) #relu output

	
	model = Model(input=xin,output=[xout_diam, xout_lob, xout_spic, xout_malig])
	
	if input_shape[1] == 32:
		lr_start = .01
	elif input_shape[1] == 64:
		lr_start = .003
	elif input_shape[1] == 128:
		lr_start = .002
	# elif input_shape[1] == 96:
		# lr_start = 5e-4
	
	opt = Nadam(lr_start,clipvalue=1.0)
	print 'compiling model'

	model.compile(optimizer=opt,loss='mse',loss_weights={'o_diam':0.06, 'o_lob':0.5, 'o_spic':0.5, 'o_mal':1.0})
	return model
	
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
	
def log_abs_error(y_true, y_pred):
	return -1 * K.mean(K.log(K.clip(1.0 - K.abs(y_true - y_pred), K.epsilon(), None)))
	
def get_generator_static(X1,IX1,X2,IX2,batch_size,augment=True):
	
	df = pd.read_csv(r"/home/ec2-user/data/annotations_enhanced.csv")
	Ydiam = df['diameter_mm'].values.astype('float32')
	# Ymargin = df['margin'].values
	Ylob = df['lobulation'].values
	Yspic = df['spiculation'].values
	Ymalig = df['malignancy'].values
	
	# print 'baseline_mse', current_frac * Y1.var() + (1 - current_frac) * Y2.var()
	current_frac = 0.9
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
		IXbatch_eq_neg1 = (IXbatch == -1)
		
		#normalize
		Xbatch = np.expand_dims(Xbatch, 1)
		if augment:
			Xbatch = random_perturb(Xbatch)
		Xbatch = Xbatch.astype('float32')
		Xbatch = (Xbatch +1000.) / (400. + 1000.)
		Xbatch = np.clip(Xbatch,0,1)
		
		Ybatch_diam = Ydiam[IXbatch]
		Ybatch_diam[IXbatch_eq_neg1] = 0.0
		
		Ybatch_lob = Ylob[IXbatch]
		Ybatch_lob[IXbatch_eq_neg1] = 0.0
		
		Ybatch_spic = Yspic[IXbatch]
		Ybatch_spic[IXbatch_eq_neg1] = 0.0
		
		Ybatch_malig = Ymalig[IXbatch]
		Ybatch_malig[IXbatch_eq_neg1] = 0.0
		
		yield Xbatch, {'o_diam':Ybatch_diam, 'o_lob':Ybatch_lob, 'o_spic':Ybatch_spic, 'o_mal':Ybatch_malig}
	

def stage_1_lr_schedule(i):
	if i < 10:
		return np.float32(.004)
	if i < 15:
		return np.float32(.002)
	# if i < 20:
		# return np.float32(.0005)
	# if i < 23:
		# return np.float32(.0001)
	return np.float32(3e-5)
	
def fine_tune_lr_schedule(i):
	if i == 0:
		return 3e-4
	if i == 1:
		return 1e-4
	if i == 2:
		return 3e-5
	return 3e-5 
		
def stage_2_lr_schedule(i):
	if i < 15:
		return np.float32(.002)
	if i < 20:
		return np.float32(.0006)
	if i < 23:
		return np.float32(.0002)
	return np.float32(3e-5)
	
def get_batch_size_for_stage(stage):
	
	#return np.around( 64. * (64 ** 3) / (stage ** 3))
	if stage == 32:
		batch_size=128
	if stage == 64:
		batch_size=64
	# if stage == 128:
		# batch_size=8
	# if stage == 256:
		# batch_size=2
	# if stage == 72:
		# batch_size=40
	# if stage == 65:
		# batch_size=63
	# if stage == 96:
		# batch_size=18
	return batch_size
	
	
def train_model_on_stage(stage,model,fast_start=False,split=True,fine_tune=False):
	test_mode=False
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
		Xpos = np.load(r"/home/ec2-user/data/Xpositive_temp_v5_" + str(stage) + ".npy")
		IXpos = np.load(r"/home/ec2-user/data/IXpositive_temp_v5_" + str(stage) + ".npy")
		Xpos = np.roll(Xpos,axis=0,shift=int(.5*Xpos.shape[0]))
                IXpos = np.roll(IXpos, axis=0, shift=int(.5*Xpos.shape[0]))

                split_pos = int(.75*Xpos.shape[0])
		Xtrain_pos = Xpos[:split_pos]
		Xvalid_pos = Xpos[split_pos:]
		IXtrain_pos = IXpos[:split_pos]
		IXvalid_pos = IXpos[split_pos:]
		del Xpos, IXpos
		
		Xneg = np.load(r"/home/ec2-user/data/Xrandom_temp_v5_" + str(stage) + ".npy")
		IXneg = np.load(r"/home/ec2-user/data/IXrandom_temp_v5_" + str(stage) + ".npy")
	        Xneg = np.roll(Xneg,axis=0,shift=int(.5*Xneg.shape[0]))
                IXneg = np.roll(IXneg, axis=0, shift=int(.5*Xneg.shape[0]))

	        split_neg = int(.75*Xneg.shape[0])
		Xtrain_neg = Xneg[:split_neg]
		Xvalid_neg = Xneg[split_neg:]
		IXtrain_neg = IXneg[:split_neg]
		IXvalid_neg = IXneg[split_neg:]
		del Xneg, IXneg
		
		train_generator = get_generator_static(Xtrain_pos,IXtrain_pos, Xtrain_neg, IXtrain_neg, augment=True, batch_size=batch_size)
		valid_generator = get_generator_static(Xvalid_pos,IXvalid_pos, Xvalid_neg, IXvalid_neg,  augment=True, batch_size=batch_size)
	else:
		Xpos = np.load(r"/home/ec2-user/data/Xpositive_temp_v5_" + str(stage) + ".npy")
		IXpos = np.load(r"/home/ec2-user/data/IXpositive_temp_v5_" + str(stage) + ".npy")
		Xneg = np.load(r"/home/ec2-user/data/Xrandom_temp_v5_" + str(stage) + ".npy")
		IXneg = np.load(r"/home/ec2-user/data/IXrandom_temp_v5_" + str(stage) + ".npy")
		
		train_generator = get_generator_static(Xpos,IXpos, Xneg, IXneg, augment=True, batch_size=batch_size)

				
	name = 'model_des_v38_mse_' + str(stage) + '_{epoch:02d}.h5'
	chkp = ModelCheckpoint(filepath=name)

	if stage == 32:
		lr_schedule = LearningRateScheduler(stage_1_lr_schedule)
		nb_epoch = 15
		samples_per_epoch = 150
	else:
		lr_schedule = LearningRateScheduler(stage_2_lr_schedule)
		nb_epoch=25
		samples_per_epoch = 200
		
	print 'restarting data gen process'
	data_gen_process = Process(target=data_generator_fn.main,args=[stage,np.random.randint(0,10000)])
	data_gen_process.start()
	
	for epoch in range(nb_epoch):
		#check after each epoch if the data is done and if so reload it
		if not data_gen_process.is_alive():
			#reload data
			print 'RELOADING DATA'
			if split:
				Xpos = np.load(r"/home/ec2-user/data/Xpositive_temp_v5_" + str(stage) + ".npy")
		                IXpos = np.load(r"/home/ec2-user/data/IXpositive_temp_v5_" + str(stage) + ".npy")
		                Xpos = np.roll(Xpos,axis=0,shift=int(.5*Xpos.shape[0]))
                                IXpos = np.roll(IXpos, axis=0, shift=int(.5*Xpos.shape[0]))

                                Xplit_pos = int(.75*Xpos.shape[0])
				Xtrain_pos = Xpos[:split_pos]
				Xvalid_pos = Xpos[split_pos:]
				IXtrain_pos = IXpos[:split_pos]
				IXvalid_pos = IXpos[split_pos:]
				del Xpos, IXpos
				
				Xneg = np.load(r"/home/ec2-user/data/Xrandom_temp_v5_" + str(stage) + ".npy")
		                IXneg = np.load(r"/home/ec2-user/data/IXrandom_temp_v5_" + str(stage) + ".npy")
		                Xneg = np.roll(Xneg,axis=0,shift=int(.5*Xneg.shape[0]))
                                IXneg = np.roll(IXneg, axis=0, shift=int(.5*Xneg.shape[0]))

                                split_neg = int(.75*Xneg.shape[0])
				Xtrain_neg = Xneg[:split_neg]
				Xvalid_neg = Xneg[split_neg:]
				IXtrain_neg = IXneg[:split_neg]
				IXvalid_neg = IXneg[split_neg:]
				del Xneg, IXneg
		
				train_generator = get_generator_static(Xtrain_pos,IXtrain_pos, Xtrain_neg, IXtrain_neg, augment=True, batch_size=batch_size)
				valid_generator = get_generator_static(Xvalid_pos,IXvalid_pos, Xvalid_neg, IXvalid_neg,  augment=True, batch_size=batch_size)
			else:
				Xpos = np.load(r"/home/ec2-user/data/Xpositive_temp_v5_" + str(stage) + ".npy")
		                IXpos = np.load(r"/home/ec2-user/data/IXpositive_temp_v5_" + str(stage) + ".npy")
		                Xneg = np.load(r"/home/ec2-user/data/Xrandom_temp_v5_" + str(stage) + ".npy")
                		IXneg = np.load(r"/home/ec2-user/data/IXrandom_temp_v5_" + str(stage) + ".npy")
		
				train_generator = get_generator_static(Xpos,IXpos, Xneg, IXneg, augment=True, batch_size=batch_size)

			
			
			#restart data generator
			data_gen_process = Process(target=data_generator_fn.main,args=[stage,np.random.randint(0,10000)])
			data_gen_process.start()
		else:
			print 'data generator still running'
			
		print 'epoch', epoch, 'model lr', model.optimizer.lr.get_value()
		if split:
			model.fit_generator(train_generator,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[chkp,lr_schedule],
								validation_data=valid_generator, nb_val_samples=samples_per_epoch*batch_size/2,initial_epoch=epoch)
		else:
			model.fit_generator(train_generator,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=epoch+1,callbacks=[chkp,lr_schedule],
								initial_epoch=epoch)
	
	if fine_tune:
	        Xpos = np.load(r"/home/ec2-user/data/Xpositive_temp_v5_" + str(stage) + ".npy")
	        IXpos = np.load(r"/home/ec2-user/data/IXpositive_temp_v5_" + str(stage) + ".npy")
	        Xneg = np.load(r"/home/ec2-user/data/Xrandom_temp_v5_" + str(stage) + ".npy")
                IXneg = np.load(r"/home/ec2-user/data/IXrandom_temp_v5_" + str(stage) + ".npy")
		
		train_generator = get_generator_static(Xpos,IXpos, Xneg, IXneg, augment=True, batch_size=batch_size)
		
		name = 'model_des_v38_mse_' + str(stage) + '_finetune_{epoch:02d}.h5'
		chkp = ModelCheckpoint(filepath=name)
		model.optimizer.lr.set_value(1e-3)
		lr_schedule = LearningRateScheduler(fine_tune_lr_schedule)
		#fit on the entire dataset for a few epochs with small learn rate
		model.fit_generator(train_generator,samples_per_epoch=samples_per_epoch*batch_size, nb_epoch=5,callbacks=[chkp,lr_schedule])
	
	return model
		
if __name__ == '__main__':

	from keras.models import Model, load_model
	from keras.layers import Input, Lambda, Dense, Flatten, Reshape, merge, Highway, Activation,Dropout
	from keras.layers.convolutional import Convolution3D
	from keras.layers.pooling import GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D,GlobalAveragePooling3D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.noise import GaussianDropout
	from keras.optimizers import Adamax, Adam, Nadam
	from keras.layers.advanced_activations import ELU,PReLU,LeakyReLU
	from keras import backend as K
	from keras.regularizers import l2
	from keras.models import load_model
	from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler, CSVLogger
	import numpy as np # linear algebra
	import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
	import os
	from scipy import ndimage
	import data_generator_fn3 as data_generator_fn
	from multiprocessing import Process
	np.random.seed(23122)
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
	
	#v37_mse_describer 64 model: 4.5 train loss and 4.27 valid loss @ 15
	
	#so clearly the v37_mse_describer model isn't as good as the v24. 
	#the major changes here were removing the branching and replacing it with 1x1 convs.
	
	
	model_32 = build_model((1,32,32,32))
	model_64 = build_model((1,64,64,64))
	# model_64 = load_model("F:\Flung\descriptor models\model_des_v38_mse_64_08.h5")
	# model_96 = build_model((1,96,96,96))
	 # model_256 = build_model((1,256,256,256))
	
	model_32.summary()
	
	# model_32.load_weights('model_v38_mse_describer_weights_temp.h5')
	#print 'prior model (v34 desc) obtained malig MAE train/val 0.357613328172 0.454452694203'
	print 'prior model ended at 1.157/1.5 ish train/val @ 25'
	print 'v37b ended at 1.0/1.15 trn/val @ 25'
	model_32 = train_model_on_stage(32, model_32,fast_start=False,split=True)
	# #now save these weights and reload them in the next model
	model_32.save_weights('model_v38_mse_describer_weights_temp.h5')
	model_64.load_weights('model_v38_mse_describer_weights_temp.h5')
	# model_64 = load_model("model_des_v38_mse_64_12.h5")
	model_64 = train_model_on_stage(64, model_64,fast_start=False,split=True,fine_tune=True)
	model_64.save_weights('model_v38_mse_describer_weights_temp64.h5')

	
	print 'done.'
