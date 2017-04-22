from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor as XTR
import numpy as np
class SlightlyNonlinearClassification(BaseEstimator):
	
	def __init__(self,C=10,step_size=0.7):
		# self.lr_params = lr_params
		# self.xtr_params = xtr_params
		self.step_size = step_size
		self.C = C
		
	def fit(self, X, y):
		lr = LogisticRegression(C=self.C,penalty='l1')
		lr_preds = cross_val_predict(lr, X, y, cv=50, method='predict_proba')[:,1]
		lr.fit(X,y)
		#CalibratedClassifierCV(LogisticRegression(**lr_params), method='sigmoid', cv=15).fit(X,y)
		xtr = XTR(n_estimators=500, min_samples_leaf=20, max_features=.8).fit(X, y - lr_preds)
		self.lr = lr
		self.xtr = xtr
		
		return self
	
	def predict_proba(self, X):
		base = self.lr.predict_proba(X)
		base[:,1] += self.step_size * self.xtr.predict(X)
		base[:,0] = 1 - base[:,1]
		return np.clip(base, 1e-3, 1-1e-3)
		
	def predict(self, X):
		return self.predict_proba(X)

		
def process_ens2_sg1(names,labels):
	ens2_models = []
	ens2_columns = []

	df_masses = pd.read_csv(r'F:\Flung\stage2\ensembling\stage1_masses_predictions.csv')
	df_masses = df_masses.rename(columns={'patient_id':'id', 'prediction':'mass_pred'})

	dfs_sg1 = []

	for name in names:
		dfs_sg1.append(pd.read_csv(r'F:\Flung\stage2\ensembling\model_features_stage1_' + name + '.csv'))
	#now we have one df per model
	#create a set of predictions per model
	#need to join on the target for this.

	#we have the targets.
	df_ens2 = df_masses[['id']]
	
	for i,(df,name) in enumerate(zip(dfs_sg1,names)):
		df['id'] = df['patient'].apply(lambda x: x.split('_')[0])
		df = pd.merge(left=labels, right=df, how='outer',left_on='id',right_on='id')
		df = pd.merge(left=df, right=df_masses,how='outer',on='id')
		
		train_filter = pd.notnull(df['cancer']).values
		# df.to_csv('test.csv')
		x_cols = df.drop(['id', 'cancer', 'patient'],1).columns
		X = df.loc[train_filter][x_cols].values
		Y = df.loc[train_filter]['cancer'].values
		# print X.shape, Y.shape
		# lr = CalibratedClassifierCV(LogisticRegression(penalty='l1', C=10), method='sigmoid',cv=10)
		# lr = LogisticRegression(penalty='l1', C=10000)
		lr = SlightlyNonlinearClassification(C=10000, step_size=0.9)
		Yh = cross_val_predict(lr, X, Y, cv=25, method='predict_proba',n_jobs=5)[:,1]
		df.loc[train_filter, 'yh_' + name] = Yh
		lr.fit(X,Y)
		
		ens2_models.append(lr)
		ens2_columns.append(x_cols)
		
		Xtest = df.loc[~train_filter][x_cols].values
		if Xtest.shape[0] > 0:
			df.loc[~train_filter,'yh_' + name] = lr.predict_proba(Xtest)[:,1]
		else:
			print 'found no missing labels for ens2 stage1. does this make sense?'
			
		df = df[['id', 'yh_' + name]]
		
		df_ens2 = pd.merge(df_ens2, df, how='outer', on='id')
		print names[i]
		
	#now we have oos predictions for each sample. Let's construct our blended solution...

	
	df_ens2 = df_ens2.set_index('id')
	df_ens2['yh_ens2'] = np.mean([df_ens2[c] for c in df_ens2.columns if c[:2] == 'yh'],axis=0)
	
	labels2 = pd.merge(labels,df_ens2, how='left', left_on='id',right_index=True)
	print 'log loss ens2', log_loss(labels2['cancer'], labels2['yh_ens2'])

	df_ens2 = df_ens2[['yh_ens2']]
	df_ens2.to_csv('ens2_preds_stage1.csv')
	return ens2_models, ens2_columns
	
	
def process_ens1_sg1(labels):
	df_ens1 = pd.read_csv(r"F:\Flung\stage2\ensembling\weighted_ensemble_v1_nodulesv29_stage1.csv")
	df_ens1['id'] = df_ens1['patient'].apply(lambda x: x.split('_')[0])
	
	df_masses = pd.read_csv(r'F:\Flung\stage2\ensembling\stage1_masses_predictions.csv')
	df_masses = df_masses.rename(columns={'patient_id':'id', 'prediction':'mass_pred'})
	df_ens1 = pd.merge(left=df_ens1, right=df_masses, how='outer', on='id')
	
	df_ens1 = pd.merge(left=labels, right=df_ens1, how='outer',on='id')
	# print df_ens1.head()
	train_filter = pd.notnull(df_ens1['cancer']).values
	ens1_cols = df_ens1.drop(['id', 'cancer', 'patient'],1).columns
	X = df_ens1.loc[train_filter][ens1_cols].values
	Y = df_ens1.loc[train_filter]['cancer'].values
	# ens1_lr = LogisticRegression(penalty='l1', C=10000)
	ens1_lr = SlightlyNonlinearClassification(C=10, step_size=.2)
	#ens1_lr = CalibratedClassifierCV(LogisticRegression(penalty='l1', C=10), method='sigmoid',cv=10)
	Yh = cross_val_predict(ens1_lr, X, Y, cv=25, method='predict_proba',n_jobs=6)[:,1]
	df_ens1.loc[train_filter, 'yh_ens1'] = Yh
	ens1_lr.fit(X,Y)
	Xtest = df_ens1.loc[~train_filter][ens1_cols].values
	if Xtest.shape[0] > 0:
		df_ens1.loc[~train_filter,'yh_ens1'] = ens1_lr.predict_proba(Xtest)[:,1]
	else:
		print 'found no missing labels for ensemble 1 stage 1. does this make sense?'
		
	df_ens1.set_index('id',inplace=True)
	df_ens1 = df_ens1[['yh_ens1']]
	df_ens1.to_csv('ens1_preds_stage1.csv')
	labels2 = pd.merge(labels,df_ens1, how='left', left_on='id',right_index=True)
	print 'log loss ens1', log_loss(labels2['cancer'], labels2['yh_ens1'])
	return ens1_lr, ens1_cols
	
def process_ens2_sg2(names, ens2_models, ens2_columns):
	dfs_sg2 = []
	for name in names:
		dfs_sg2.append(pd.read_csv(r'F:\Flung\stage2\ensembling\model_features_stage2_' + name + '.csv'))
		
	stg2_masses = pd.read_csv(r"F:\Flung\stage2\ensembling\stage2_masses_predictions.csv")
	stg2_masses = stg2_masses.rename(columns={'patient_id':'id', 'prediction':'mass_pred'})
	
	df_stg2_ens2 = stg2_masses[['id']]
	
	for i, dfi, modeli, namei, colsi in zip(range(7), dfs_sg2, ens2_models, names, ens2_columns):
		#score the model on this df
		dfi = dfi.rename(columns={dfi.columns[0]:'patient'})
		dfi['id'] = dfi['patient'].apply(lambda x: x.split('_')[0])
		dfi = pd.merge(dfi, stg2_masses, how='outer', on='id')
		X = dfi[colsi].values
		Yh = modeli.predict_proba(X)[:,1]
		dfi['yh_' + namei] = Yh
		dfi = dfi[['id', 'yh_' + namei]]
		df_stg2_ens2 = pd.merge(df_stg2_ens2, dfi, how='outer', on='id')
		
	#now we have predictions for each model, average.
	df_stg2_ens2['yh_ens2'] = np.mean([df_stg2_ens2[c] for c in df_stg2_ens2.columns if c[:2] == 'yh'],axis=0)
	df_stg2_ens2 = df_stg2_ens2[['id', 'yh_ens2']]
	df_stg2_ens2.to_csv('ens2_preds_stage2.csv', index=False)
	
	
def process_ens1_sg2(ens1_model, ens1_cols):
	df_ens1 = pd.read_csv(r"F:\Flung\stage2\ensembling\weighted_ensemble_v1_nodulesv29_stage2.csv")
	df_ens1['id'] = df_ens1['patient'].apply(lambda x: x.split('_')[0])
	
	df_masses = pd.read_csv(r'F:\Flung\stage2\ensembling\stage2_masses_predictions.csv')
	df_masses = df_masses.rename(columns={'patient_id':'id', 'prediction':'mass_pred'})
	df_ens1 = pd.merge(left=df_ens1, right=df_masses, how='outer', on='id')
	
	Xtest = df_ens1[ens1_cols].values
	Yh = ens1_model.predict_proba(Xtest)[:,1]
	df_ens1['yh_ens1'] = Yh
	
	df_ens1.set_index('id',inplace=True)
	df_ens1 = df_ens1[['yh_ens1']]
	df_ens1.to_csv('ens1_preds_stage2.csv')
	
	
def process_jul():
	#after stage 2 starts 
	#julian will update the 'train' files so that they contain the full stage1 dataset
	#and he will update the 'test' files so that they contain the stage2 dataset
	jul_columns = [u'mask_size', u'mass', u'mx_10', u'ch_10', u'cnt_10',
		   u'med_10', u'wmx_10', u'crdz_10', u'mx2_10', u'crdy_10', u'crdx_10',
		   u'mx_15', u'ch_15', u'cnt_15', u'med_15', u'wmx_15', u'crdz_15',
		   u'mx2_15', u'crdy_15', u'crdx_15', u'mx_20', u'ch_20', u'cnt_20',
		   u'med_20', u'wmx_20', u'crdz_20', u'mx2_20', u'crdy_20', u'crdx_20']
	
	jul_fs_trn = pd.read_csv(r"F:\Flung\stage2\ensembling\train_luna16_fs.csv")
	jul_fs_test = pd.read_csv(r"F:\Flung\stage2\ensembling\submission_luna16_fs.csv")
	assert all(jul_fs_trn.columns == jul_fs_test.columns)

	Xfs = jul_fs_trn[jul_columns].values
	Yfs = jul_fs_trn['cancer_label'].values
	xt_fs = XT(n_estimators=500, min_samples_leaf=20, max_features=.9)
	Yh_fs = cross_val_predict(xt_fs, Xfs, Yfs, cv=20, method='predict_proba', n_jobs=5)[:,1]
	xt_fs.fit(Xfs, Yfs)
	
	jul_fs_trn['yh_fs'] = Yh_fs
	Xfs_test = jul_fs_test[jul_columns].values
	Yh_fs_test = xt_fs.predict_proba(Xfs_test)[:,1]
	jul_fs_test['yh_fs'] = Yh_fs_test
	# jul_fs = pd.concat([jul_fs_trn, jul_fs_test],1)
	
	jul_dsb1_trn = pd.read_csv(r"F:\Flung\stage2\ensembling\train_luna_posnegndsb_v1.csv")
	jul_dsb1_test = pd.read_csv(r"F:\Flung\stage2\ensembling\submission_luna_posnegndsb_v1.csv")
	assert all(jul_fs_trn.columns == jul_fs_test.columns)
	Xdsb1 = jul_dsb1_trn[jul_columns].values
	Ydsb1 = jul_dsb1_trn['cancer_label'].values
	xt_dsb1 = XT(n_estimators=500, min_samples_leaf=20, max_features=.9)
	Yh_dsb1 = cross_val_predict(xt_dsb1, Xdsb1, Ydsb1, cv=20, method='predict_proba', n_jobs=5)[:,1]
	xt_dsb1.fit(Xdsb1, Ydsb1)
	
	jul_dsb1_trn['yh_dsb1'] = Yh_dsb1
	Xdsb1_test = jul_dsb1_test[jul_columns].values
	Yh_dsb1_test = xt_dsb1.predict_proba(Xdsb1_test)[:,1]
	jul_dsb1_test['yh_dsb1'] = Yh_dsb1_test
	# jul_dsb1 = pd.concat([jul_dsb1_trn, jul_dsb1_test],1)
	
	jul_dsb2_trn = pd.read_csv(r"F:\Flung\stage2\ensembling\train_luna_posnegndsb_v2.csv")
	jul_dsb2_test = pd.read_csv(r"F:\Flung\stage2\ensembling\submission_luna_posnegndsb_v2.csv")
	assert all(jul_fs_trn.columns == jul_fs_test.columns)
	Xdsb2 = jul_dsb2_trn[jul_columns].values
	Ydsb2 = jul_dsb2_trn['cancer_label'].values
	xt_dsb2 = XT(n_estimators=500, min_samples_leaf=20, max_features=.9)
	Yh_dsb2 = cross_val_predict(xt_dsb2, Xdsb2, Ydsb2, cv=20, method='predict_proba', n_jobs=5)[:,1]
	xt_dsb2.fit(Xdsb2, Ydsb2)
	
	jul_dsb2_trn['yh_dsb2'] = Yh_dsb2
	Xdsb2_test = jul_dsb2_test[jul_columns].values
	Yh_dsb2_test = xt_dsb2.predict_proba(Xdsb2_test)[:,1]
	jul_dsb2_test['yh_dsb2'] = Yh_dsb2_test
	# jul_dsb2 = pd.concat([jul_dsb2_trn, jul_dsb2_test],1)
	
	#one time: build stacker on these and figure out optimal weights
	# print jul_fs.columns

	#export the train and the test predictions
	jul_trn = jul_fs_trn[['patient_id', 'yh_fs']]
	jul_trn = pd.merge(left=jul_trn, right=jul_dsb1_trn[['patient_id', 'yh_dsb1']], how='outer', on='patient_id')
	jul_trn = pd.merge(left=jul_trn, right=jul_dsb2_trn[['patient_id', 'yh_dsb2']], how='outer', on='patient_id')
	jul_trn['yh_jul'] = 0.5 * jul_trn['yh_fs'] + 0.25 * jul_trn['yh_dsb1']  + 0.25 * jul_trn['yh_dsb2']
	jul_trn = jul_trn[['patient_id', 'yh_jul']]
	jul_trn.to_csv('julian_preds_train.csv',index=False)
	
	jul_test = jul_fs_test[['patient_id', 'yh_fs']]
	jul_test = pd.merge(left=jul_test, right=jul_dsb1_test[['patient_id', 'yh_dsb1']], how='outer', on='patient_id')
	jul_test = pd.merge(left=jul_test, right=jul_dsb2_test[['patient_id', 'yh_dsb2']], how='outer', on='patient_id')
	jul_test['yh_jul'] = 0.5 * jul_test['yh_fs'] + 0.25 * jul_test['yh_dsb1']  + 0.25 * jul_test['yh_dsb2']
	jul_test = jul_test[['patient_id', 'yh_jul']]
	jul_test.to_csv('julian_preds_test.csv',index=False)
	
	
ENS_FILES_DIR = r'F:\Flung\stage2\ensembling'

if __name__ == '__main__':

	import pandas as pd
	import numpy as np
	from sklearn.linear_model import LogisticRegression
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.model_selection import cross_val_predict, StratifiedKFold
	from sklearn.metrics import log_loss
	from sklearn.ensemble import ExtraTreesClassifier as XT
	np.random.seed(42)

	labels = pd.read_csv(r"F:\Flung\stage2\stage1plus2_labels.csv")
	#expect as input 7 data frames
	names = ['37', '37b', '37c', '37d', '37f', '37g', '38']
	
	process_jul()
	
	ens2_models, ens2_columns = process_ens2_sg1(names, labels)
	
	ens1_model, ens1_cols = process_ens1_sg1(labels)
	
	process_ens2_sg2(names, ens2_models, ens2_columns)

	process_ens1_sg2(ens1_model, ens1_cols)
	

		