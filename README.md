# Gesture Recognition

A deep learning project that classifies hand-gesture videos into 5 commands, aimed at a "control your smart TV without a remote" use case.

## What this covers
Video classification with a CNN + ConvLSTM2D architecture in Keras/TensorFlow: given a short video (30 frames) of a hand gesture, predict which of 5 gestures it is — Thumbs Up (volume up), Thumbs Down (volume down), Left Swipe (rewind 10s), Right Swipe (forward 10s), or Stop (pause). The notebook works through several architecture experiments (Conv3D, TimeDistributed-Conv2D+GRU, TimeDistributed-Conv2D+ConvLSTM2D) before settling on a final model.

## Tech Stack
- Python, Jupyter Notebook
- **Keras / TensorFlow** — model definition and training (`Sequential`, `Conv2D`, `ConvLSTM2D`, `TimeDistributed`, `BatchNormalization`, `GlobalAveragePooling2D`)
- **NumPy / PIL** — frame loading, resizing, and preprocessing via a custom generator
- **Matplotlib** — training/validation loss & accuracy curves

## Approach
- **Data:** train/validation sets of gesture videos, each stored as a folder of 30 image frames at one of two resolutions (360x360 or 120x160); `train.csv`/`val.csv` map each folder to a gesture label (0-4).
- **Preprocessing:** custom data generator resizes frames to 120x120 and normalizes pixel values to [0, 1]; augmentation (edge enhancement, Gaussian blur, brightness adjustment) applied during training.
- **Architecture search:** iterated through multiple candidate models (documented as numbered experiments in the notebook) — plain Conv3D, TimeDistributed Conv2D + GRU, and TimeDistributed Conv2D + ConvLSTM2D — before selecting the final one below.
- **Final model:** `TimeDistributed(Conv2D)` layers for per-frame feature extraction → `BatchNormalization` → `ConvLSTM2D` to capture motion/temporal information across the frame sequence → `GlobalAveragePooling2D` + `Dense` layers → softmax over the 5 gesture classes.
- **Training:** Adam optimizer, categorical cross-entropy loss, `ReduceLROnPlateau` for learning-rate decay, and `ModelCheckpoint` saving a `.h5` file after every epoch (see `model-00049-0.06883-0.96726-0.13135-0.93750.h5` in this repo).

## Results
The checkpoint included in this repo (`model-00049-0.06883-0.96726-0.13135-0.93750.h5`, epoch 49) reports:
- Training accuracy: **96.7%** (loss 0.069)
- Validation accuracy: **93.75%** (loss 0.131)

## Running Locally
```bash
git clone https://github.com/vinay23is/Gesture_Recognition.git
cd Gesture_Recognition
pip install numpy tensorflow keras matplotlib opencv-python pillow jupyter
jupyter notebook "Gesture Recognition.ipynb"
```
The training/validation video data is not included in this repo (it's a large dataset of per-frame images) — [download it here](https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL). The notebook expects it laid out with a `train.csv`/`val.csv` alongside the frame folders. The included `.h5` file is a trained checkpoint that can be loaded directly with `keras.models.load_model(...)` for inference without retraining.
