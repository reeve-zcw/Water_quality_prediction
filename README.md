# Water quality prediction
This repository contains a project that predicts the values of various substances in water quality using weather information. This README explains how to run the project and important considerations.

## Getting Started
To get started with this project, please follow the steps below:

1. Clone the repository: `git clone https://github.com/reeve-zcw/Water_quality_prediction.git`
2. Install the required dependencies using following code:
```
pip install -r requirement.txt
```

## Usage
To use this project, execute `main.py`. During execution, you can visualize the training process, and the trained weights will be saved in the model_weights" directory.

## Visualization
To visualize the changes in the loss function, model architecture, and other information during model training using Tensorboard, follow these steps:

1. Navigate to the project directory using the terminal.
2. Enter and execute the following code:
```
tensorboard --logdir="logs"
```
You also can replace "logs" with a different name if desired, but make sure to update line 75 of `main.py` with the corresponding folder name.

3. Click on the provided URL, `http://localhost:6006/`, in the terminal.
4. Execute `main.py` to train the model.
5. Click the __SCALARS__ section in `http://localhost:6006/` to view the model's training process.
6. Click the __GRAPHS__ section in `http://localhost:6006/` to view the model's structure.

## Notes
1. There are two versions of the dataset: one using weather information from the same day as water quality testing and another using weather information from the previous day. The current implementation in `main.py` uses the latter.
2. After testing, it was found that the trained model performs with high accuracy when predicting data it has been trained on, but it shows larger errors when dealing with new data. This is likely due to overfitting caused by a small dataset.
