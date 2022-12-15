To install code, create virtual environment with 3.7.11 and follow readme in main directory.

To train: Have training and validaiton data in /dataset/train and /dataset/val respectively. Each folder should contain all images and JSON file created using VGG with bounding boxes in the labels folder. Run "python training.py" from root.

To test: In predicition.py modify line 38 to have path to image to test on, default is /dataset/test/. Run "python training.py" from root. Intermediary result and final result will be displayed, final result will be saved to /out/.
