import os
os.environ['KAGGLE_USERNAME'] = "NAME" # username from the json file
os.environ['KAGGLE_KEY'] = "KAGGLE_KEY" # key from the json file
!kaggle competitions download -c facial-keypoints-detection
!unzip test.zip
!unzip training.zip
