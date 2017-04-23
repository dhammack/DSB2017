# 2nd Place Solution To the 2017 National Data Science Bowl

This is my contribution to the 2017 NDSB 2nd place team solution. The other half is located here https://github.com/juliandewit/kaggle_ndsb2017

For documentation and technical details, see the file here: https://github.com/dhammack/DSB2017/blob/master/dsb_2017_daniel_hammack.pdf.

scoring_code: location of code which replicates the submission for stage 2 of the competition. Trained models will eventually go here once I have checked that it is OK to upload them. You will not be able to run this code until the models have been uploaded.

training_code: location of the code to rebuild the models required for scoring my part of the solution.

### NOTE

Most, if not every, python script in this repo currently use absolute filepaths when referring to files. This was much easier for me to write originally but will cause issues when trying to replicate. If you are trying to rebuild/rescore my solution make sure to check the filepaths.

Also I sometimes make modifications to my local Keras install to try out new things. I'm planning on going over my code to check for these but I haven't done it yet. If you get a strange error where my code is trying to use a feature in Keras that doesn't exist, this is probably what happened. As far as I can recall the only times this should happen are custom initializations (orthogonal and looks-linear are the two that I may have done this with) and custom activations (I don't think I used any of these...).


Also - I have noticed that a newer version of OpenCV can break some of my code. If you get OpenCV errors, change the following line:

```
contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

```
To:

```
_, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

```

### REQUIRED DATA

for training: 
LUNA16 - neural net models were trained only on this data. note you will need the annotations_enhanced.csv file included in this repo which contains LIDC radiologist annotations for the LUNA16 nodules.

NDSB 2017 stage1 data - used for training final diagnosis model (not a neural network)

for scoring:

any dataset of DICOM files.


### Actually using this

If you are interested in actually using this code in a real application rather than just replicating my work, please reach out. This code is unnecessarily complicated due to the cutthroat and very hasty nature of Kaggle competitions. It could be considerably simplified and sped up with no loss in performance. Furthermore, now that the competition is over and I have time to think clearly, I know of several ways to improve the performance of this system. 



TODO: this readme.

