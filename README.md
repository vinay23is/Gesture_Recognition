# Gesture Recognition using Deep Learning

## Project Overview
This project focuses on building a **3D Convolutional Neural Network (3D-CNN)** to recognize five different hand gestures. The model is trained on a dataset of videos, each containing 30 frames, to accurately classify user gestures. This system is intended for **smart TVs** where users can control functions like increasing/decreasing volume and skipping forward/backward using hand gestures.

## Problem Statement
Imagine working as a data scientist at a **home electronics company** that manufactures smart TVs. You aim to develop a feature that enables users to **control the TV without a remote** using hand gestures. A webcam continuously monitors gestures, and each gesture corresponds to a specific command:

- 👍 **Thumbs Up** → Increase Volume  
- 👎 **Thumbs Down** → Decrease Volume  
- 👈 **Left Swipe** → Jump Backward 10 seconds  
- 👉 **Right Swipe** → Jump Forward 10 seconds  
- ✋ **Stop** → Pause the Movie  

## Dataset  
The dataset consists of a **train** and **validation** set, each containing multiple videos stored as image frames. Each **video folder contains 30 images** of a specific hand gesture. The dataset includes two image resolutions:
- **360x360**
- **120x160**

### Dataset Access
Download the dataset from the following link:  
[Dataset Link](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL)

### CSV File Structure
Each row in `train.csv` and `val.csv` corresponds to a **video** and contains:
- **Folder name** (contains 30 images)
- **Gesture name**
- **Label (0-4)** representing one of the five gestures

## Model Architecture
We experimented with multiple deep learning models, including **Conv3D**, **ConvLSTM2D**, and **TimeDistributed Conv2D with GRU**. The final model uses a **ConvLSTM2D** architecture for improved accuracy.

### Final Model:
- **TimeDistributed Conv2D Layers** for feature extraction  
- **Batch Normalization** to improve convergence  
- **ConvLSTM2D Layer** to capture spatio-temporal information  
- **Global Average Pooling & Dense Layers** for classification  
- **Softmax Activation** for predicting one of the five gestures  

## Training Strategy
- **Augmentation:** Applied techniques like edge enhancement, Gaussian blur, and brightness modification.  
- **Normalization:** Images were resized to **120x120** and pixel values were scaled between **0-1**.  
- **Optimizer:** Adam with learning rate decay using ReduceLROnPlateau.  
- **Loss Function:** Categorical Crossentropy.  

## Model Performance
- Achieved **96% training accuracy** and **95% validation accuracy**.
- Optimized the model to use **minimum parameters** for real-time performance on edge devices.

## Installation & Dependencies
1. Install the required libraries:
   ```bash
   pip install numpy tensorflow keras matplotlib opencv-python pillow
