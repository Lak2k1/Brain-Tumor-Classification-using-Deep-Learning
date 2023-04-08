# Brain Tumor Classification using Deep Learning
## Brain Tumor Classification using Deep Learning
This repository contains the code and dataset for a deep learning model that can classify brain tumor images into different categories. The model uses Convolutional Neural Networks (CNNs), Convolution 2D, Batch Normalization, Max Pooling, and Dropout techniques for optimal performance.

We trained the model on pre-trained networks such as VGG16, VGG19, and ResNet50, but in the end, we developed a better model that achieved higher accuracy, F1 score, precision, and recall. We also used the ImageDataGenerator function to overcome the exhausted memory error.

## Dataset
We used the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle for training and testing our model.This dataset contains 7023 images of human brain MRI images which are classified into 4 classes: glioma ,meningioma ,no tumor and pituitary.

## Model Architecture
Our model consists of a sequence of convolutional, pooling, and dropout layers, followed by fully connected layers for classification. The architecture of our model is as follows:
![](https://github.com/Lak2k1/Brain-Tumor-Classification-using-Deep-Learning/blob/main/model%20architecture.png)
