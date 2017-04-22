if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	from sklearn.linear_model import LogisticRegression
	from sklearn.calibration import CalibratedClassifierCV
	from sklearn.model_selection import cross_val_predict, StratifiedKFold
	from sklearn.metrics import log_loss
	from sklearn.ensemble import ExtraTreesClassifier as XT

	np.random.seed(42)

	#TODO: Replace stage1 with stage2
	stage2 = True
	if not stage2:
	
		labels = pd.read_csv(r"E:\lung\stage1_labels.csv")
		
		dh_ens1 = pd.read_csv(r"F:\Flung\ensembling\ens1_preds_stage1.csv")
		dh_ens2 = pd.read_csv(r"F:\Flung\ensembling\ens2_preds_stage1.csv")

		jul_preds_trn = pd.read_csv("F:\Flung\ensembling\julian_preds_train.csv")
		# jul_preds_test = pd.read_csv("F:\Flung\ensembling\julian_preds_test.csv")
		
		labels = pd.merge(labels, dh_ens1, how='inner', on='id')
		labels = pd.merge(labels, dh_ens2, how='inner', on='id')
		labels = pd.merge(labels, jul_preds_trn, how='inner', left_on='id', right_on='patient_id')
		
		
		from sklearn.linear_model import Ridge
		
		r = Ridge(alpha=.001, fit_intercept=False)
		print np.corrcoef(labels['yh_ens1'], labels['yh_ens2'])
		Xmeta = labels[['yh_jul', 'yh_ens1', 'yh_ens2']].values
		Ymeta = labels['cancer'].values
		print 'log loss with current weights', log_loss(Ymeta, labels['yh_jul'] * .4 + .6 * (labels['yh_ens2'] * .9 + labels['yh_ens1']*.1)) 
		r.fit(Xmeta, Ymeta)
		print r.coef_
	else:
		dh_ens1 = pd.read_csv(r"F:\Flung\stage2\ensembling\ens1_preds_stage2.csv")
		dh_ens2 = pd.read_csv(r"F:\Flung\stage2\ensembling\ens2_preds_stage2.csv")
		jul_preds_test = pd.read_csv("F:\Flung\stage2\ensembling\julian_preds_test.csv")
		df = pd.merge(left=jul_preds_test, right=dh_ens1, how='inner', left_on='patient_id', right_on='id')
		df = pd.merge(left=df, right=dh_ens2, how='inner', left_on='patient_id', right_on='id')
		df['cancer'] = 0.4 * df['yh_jul'] + 0.6 * (0.7 * df['yh_ens1'] + 0.3 * df['yh_ens2'])
		df['id'] = df['patient_id']
		df = df[['id', 'cancer']]
		df.to_csv('final_predictions_dh_blend.csv',index=False)
		
		
		

	
	