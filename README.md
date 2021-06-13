# Real-Time-Facial-Expression-Recognition
This repository contains my solution to the Real-Time Facial Expression Recognition problem. The dataset used for this problem is the [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset and for data annotation of the [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset, the [FERPlus](https://github.com/microsoft/FERPlus) dataset is used. The CNN architecture used in this repository is similar to the [VGG16](https://arxiv.org/pdf/1409.1556.pdf) but the model I used has less trainable params(1.2M) compared to the original VGG16(138.4M) model and the model I used also have BatchNormalization layers for regularization purposes.

### Install
This project requires **Python** and the following Python libraries installed:
- [NumPy](https://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [shutil](https://docs.python.org/3/library/shutil.html)
- [split-folders](https://pypi.org/project/split-folders)
- [glob](https://docs.python.org/3/library/glob.html)
- [os](https://docs.python.org/3/library/os.html)
- [csv](https://docs.python.org/3/library/csv.html)
- [Pillow](https://pypi.org/project/Pillow/2.2.1)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/) distribution of Python, which already has most of the above packages. 
