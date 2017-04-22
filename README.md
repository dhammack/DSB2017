# 2nd Place Solution To the 2017 National Data Science Bowl


scoring_code: location of code which replicates the submission for stage 2 of the competition. Trained models will eventually go here once I have checked that it is OK to upload them. You will not be able to run this code until the models have been uploaded.

training_code: location of the code to rebuild the models required for scoring my part of the solution.

### NOTE

Most, if not every, python script in this repo currently use absolute filepaths when referring to files. This was much easier for me to write originally but will cause issues when trying to replicate. If you are trying to rebuild/rescore my solution make sure to check the filepaths.

Also I sometimes make modifications to my local Keras install to try out new things. I'm planning on going over my code to check for these but I haven't done it yet. If you get a strange error where my code is trying to use a feature in Keras that doesn't exist, this is probably what happened. As far as I can recall the only times this should happen are custom initializations (orthogonal and looks-linear are the two that I may have done this with) and custom activations (I don't think I used any of these...).


TODO: this readme.

