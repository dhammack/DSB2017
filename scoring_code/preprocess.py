import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from skimage import measure, morphology
# Load the scans in given folder path
INPUT_FOLDER = r'F:\Flung\stage2\stage2'
DATA_DIR = r'F:\Flung\stage2\1mm'

def load_scan(path):
	slices = [dicom.read_file(os.path.join(path , s)) for s in os.listdir(path)]
	slices.sort(key = lambda x: x.ImagePositionPatient[2], reverse=True)
	depths = [slice.ImagePositionPatient[2] for slice in slices]
	#remove slices with acquisition numbers not = 1
	
	if len(depths) != len(set(depths)):
		#duplicated positions!
		print 'file', path, 'has duplicate ImagePositionPatients'
		slices.sort(key = lambda x: x.InstanceNumber)
		slices = [s for s in slices if s.AcquisitionNumber == 1]
		
	slice_thickness = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
	if slice_thickness == 0:
		print 'image with zero slice thickness', path
		assert False 
	ip0,ip1 = slices[0].ImagePositionPatient[:2]
	err_msg = False
	for s in slices:
		s.SliceThickness = slice_thickness
		assert s.ImagePositionPatient[0] == ip0 and s.ImagePositionPatient[1] == ip1, 'error'
		#assert s.RescaleSlope == 1, 'non 1 slope'
		#assert s.SliceLocation == s.ImagePositionPatient[2], 'weird image ' + path
		if 'SliceLocation' not in s or s.SliceLocation != s.ImagePositionPatient[2] and err_msg == False:
			print 'weird patient to QA', path
			err_msg = True
			
		orient = map(float, s.ImageOrientationPatient)
		if orient != [1, 0, 0, 0, 1, 0]:
			print 'bad orient'
			print s.ImageOrientationPatient
			print orient
			assert False
		
	return slices
	
def get_pixels_hu(scans):
	image = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in scans],axis=2).astype(np.int16)
    
	# Set outside-of-scan pixels to 0
	# The intercept is usually -1024, so air is approximately 0
	image[image < -1990] = -1000
	
	# Convert to Hounsfield units (HU)
	  
	#intercept = scans[0].RescaleIntercept
	#print image.shape, image.dtype
	#image += np.int16(intercept)
	
	return np.array(image)
	
def resample(image, scan, new_spacing=[1,1,1]):
	# Determine current pixel spacing
	spacing = map(float, (scan[0].PixelSpacing + [scan[0].SliceThickness]))
	spacing = np.array(list(spacing))
	#print spacing
	resize_factor = spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = spacing / real_resize_factor
	#print new_spacing, real_resize_factor
	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
	
	return image, new_spacing
	
def process_patient(patient):
	#read, transform, save
	# target= get_target_for(patient,tgt_lookup)
	scans = load_scan(os.path.join(INPUT_FOLDER, patient)) #matches last dimensions of px_raw
	px_raw = get_pixels_hu(scans) #voxel
	px_rescaled,_ = resample(px_raw, scans, new_spacing=[1,1,1])
	
	np.save(os.path.join(DATA_DIR, patient + '_testsg2.npy'),px_rescaled)
	
def get_target_for(patient,tgt_lookup):
	return 'testsg2'
	
if __name__ == '__main__':
	# Some constants 
	
	patients = os.listdir(INPUT_FOLDER)
	patients.sort()
	# df = pd.read_csv(r"E:\lung\stage1_labels.csv")
	# df = df.set_index('id')
	# tgt_lookup = df['cancer'].to_dict()
	
	Parallel(n_jobs=8,verbose=1)(delayed(process_patient)(patient) for patient in patients)
	#process_patient('b8bb02d229361a623a4dc57aa0e5c485', tgt_lookup)
	#process_patient('00cba091fa4ad62cc3200a657aeb957e')