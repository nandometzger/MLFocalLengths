# MLFocalLengths - Estimating the Focal Length of a Single Image

Information about the focal length with which a photo is taken might be obstructed (internet photos) or not available (vintage photos). Inferring the focal length of a photo solely from a monocular view is an ill-posed task that requires knowledge about the scale of objects and their distance to the camera - e.g. scene understanding. I trained a deep learning model to acquire such scene understanding to predict the focal length and open-source the model with this repository.
 
![test](img/stone_tagged2.gif)

Focal lengths influence the distortion of an image.
Source image credits to Reddit user [u/scyshc](https://www.reddit.com/r/photography/comments/48l8uy/a_gif_showing_why_focal_length_matters/)

# Method

I preprocessed the focal lengths of ~15k of my personal image database to convert them to 35mm equivalent using [Jeffrey Friedl's LR Plugin](http://regex.info/blog/lightroom-goodies/focal-length-sort). The images were cropped to a square shape and resampled to 256x256. Using that data, I trained an EfficientNet B4 with log-transformed labels and L1 loss, which showed a mean absolute error of 16mm on the hold-out set.

# Reproducability

Set up a python environment using ``requirements.txt``. Commands for the creation of the dataset, training, and prediction are provided in the [lauch.json file](https://github.com/nandometzger/MLFocalLengths/blob/main/.vscode/launch.json).
Training data is available upon request.

The pretrained model can be accessed [here](https://drive.google.com/file/d/16Yf8dQrIAg-k8RKcy_chRsctrhQ4yzse/view?usp=share_link).


