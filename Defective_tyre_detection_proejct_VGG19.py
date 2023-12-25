#!/usr/bin/env python
# coding: utf-8

# ### Convolutional neural network model with 19 layers deep (VGG19)
# This notebook looks at defective tyre detection. The dataset used has 2 classes: `Defective Tyre`, and `Good Tyre`
# 
# The model implemented uses a VGG19 layer for feature extraction on all the images. The features are then fed to machine learninng models like SVM & random forest for binary classification.

# In[ ]:


# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import shutil
from shutil import copyfile
import os
import random
import time

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# In[ ]:


# checking TensorFlow version and GPU usage
print('Tensorflow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())


# In[ ]:


# # Download dataset
# !wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/bn7ch8tvyp-1.zip


# In[ ]:


# # unzip the dataset
# !unzip bn7ch8tvyp-1.zip


# In[ ]:


# setting path to the main directory
main_dir = "tyre_images"

# set path to defective images
defective_dir = os.path.join(main_dir, "defective")
# set path to good images
good_dir = os.path.join(main_dir, "good")

# # Print the total number of images in each directory
# print("The total number of defective images are", len(os.listdir(defective_dir)))
# print("The total number of good images are", len(os.listdir(good_dir)))


# In[ ]:


# Data Visualization
import matplotlib.image as mpimg

# Setting the no of rows and columns
ROWS = 4
COLS = 4

# Setting the figure size
fig = plt.gcf()
fig.set_size_inches(12, 12)

# get the directory to each image file in the trainset
defective_pic = [os.path.join(defective_dir, i) for i in os.listdir(defective_dir)[:8]]
good_pic = [os.path.join(good_dir, i) for i in os.listdir(good_dir)[:8]]

# merge defective and good lists
merged_list = defective_pic + good_pic

# Plotting the images in the merged list
for i, img_path in enumerate(merged_list):
    # getting the filename from the directory
    data = img_path.split('/', 7)[6]
    # creating a subplot of images with the no. of rows and colums with index no
    sp = plt.subplot(ROWS, COLS, i+1)
    # turn off axis
    sp.axis('Off')
    # reading the image data to an array
    img = mpimg.imread(img_path)
    # setting title of plot as the filename
    sp.set_title(data, fontsize=10)
    # displaying data as image
    plt.imshow(img, cmap='gray')

plt.show()  # display the plot


# In[ ]:


# Plot class distribution
plt.figure(figsize=(10,6))
x = np.arange(2)
y = [len(os.listdir(good_dir)), len(os.listdir(defective_dir))]
plt.barh(x, y)
plt.yticks(x, ["Good", "Defective"], fontsize=10)
plt.text(y[0]+5, x[0], y[0], fontsize=8)
plt.text(y[1]+5, x[1], y[1], fontsize=8)
plt.title("Distribution of Classes in the Dataset", fontsize=14);


# ### Create the training and validation directories. Move 80% of the data to the training directory for each class.

# In[ ]:


# Create "training" and "validation" directories
training_dir = os.path.join(main_dir, "training")
validation_dir = os.path.join(main_dir, "validation")

# Create "good" and "defective" subdirectories within "training" and "validation"
for sub_dir in ["good", "defective"]:
    os.makedirs(os.path.join(training_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, sub_dir), exist_ok=True)

# List all image files in "good" and "defective" directories
good_images = os.listdir(good_dir)
defective_images = os.listdir(defective_dir)

# Define the split ratio (80% for training, 20% for validation)
split_ratio = 0.8

# Shuffle the image lists
random.shuffle(good_images)
random.shuffle(defective_images)

# Split and move images to the training and validation directories
split_index_good = int(len(good_images) * split_ratio)
split_index_defective = int(len(defective_images) * split_ratio)

# Move "good" images
for i, image in enumerate(good_images):
    if i < split_index_good:
        src = os.path.join(main_dir, "good", image)
        dest = os.path.join(training_dir, "good", image)
    else:
        src = os.path.join(main_dir, "good", image)
        dest = os.path.join(validation_dir, "good", image)
    shutil.move(src, dest)

# Move "defective" images
for i, image in enumerate(defective_images):
    if i < split_index_defective:
        src = os.path.join(main_dir, "defective", image)
        dest = os.path.join(training_dir, "defective", image)
    else:
        src = os.path.join(main_dir, "defective", image)
        dest = os.path.join(validation_dir, "defective", image)
    shutil.move(src, dest)

print("Data split and directory structure updated.")


# In[ ]:


# Define paths
training_dir = os.path.join(main_dir, "training")
validation_dir = os.path.join(main_dir, "validation")

# Create training and validation directories for the 2 classes
training_defective_dir = os.path.join(training_dir, "defective")
validation_defective_dir = os.path.join(validation_dir, "defective")
training_good_dir = os.path.join(training_dir, "good")
validation_good_dir = os.path.join(validation_dir, "good")

# Calculate total training size and test size
train_size = len(os.listdir(training_defective_dir)) + len(os.listdir(training_good_dir))
val_size = len(os.listdir(validation_defective_dir)) + len(os.listdir(validation_good_dir))

# Training and validation splits
print(f"There are {len(os.listdir(training_defective_dir))} images of defective for training")
print(f"There are {len(os.listdir(training_good_dir))} images of good for training")
print(f"There are {len(os.listdir(validation_defective_dir))} images of defective for validation")
print(f"There are {len(os.listdir(validation_good_dir))} images of good for validation")

print("Total training size", train_size)
print("Total validation size", val_size)


# In[ ]:


# Load VGG19
from tensorflow.keras.applications import VGG19

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

# Freeze VGG19 layers
# conv_base.trainable = False

conv_base.summary()


# Extract the features of the all the images in the training set and validation set and store them in variables.

# In[ ]:


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 16

def extract_features(directory, sample_count):
  '''Function to extract features from images, given the directory of the images and the sample count

  Parameters:
  -----------

  directory: str, this is the directory to the different classes of images
  sample_count: int, this is the total number of samples images

  Returns:
  --------
  features (numpy array) and the corresponding label (numpy array)

  '''
  features = np.zeros(shape=(sample_count, 4, 4, 512))  # Must be equal to the output of the convolutional base
  labels = np.zeros(shape=(sample_count))
  # Preprocess data
  generator = datagen.flow_from_directory(directory,
                                          target_size=(150, 150),
                                          batch_size = batch_size,
                                          class_mode='binary')
  # Pass data through convolutional base
  i = 0
  for inputs_batch, labels_batch in generator:
      features_batch = conv_base.predict(inputs_batch)
      features[i * batch_size: (i + 1) * batch_size] = features_batch
      labels[i * batch_size: (i + 1) * batch_size] = labels_batch
      i += 1
      if i * batch_size >= sample_count:
          break
  return features, labels

start = time.time()  # record start time
train_features, train_labels = extract_features(training_dir, train_size)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(validation_dir, val_size)
end = time.time()  # record end time
print('The execution time is:', (end-start) * 10**3, 'ms')


# In[ ]:


# Reshape training data and validation
train_features = train_features.reshape(train_size, 4*4*512)
validation_features = validation_features.reshape(val_size, 4*4*512)

# save to csv file
np.savetxt('X_train_defective_VGG.csv', train_features, delimiter=',')
np.savetxt('X_val_defective_VGG.csv', validation_features, delimiter=',')
np.savetxt('y_train_defective_VGG.csv', train_labels, delimiter=',')
np.savetxt('y_val_defective_VGG.csv', validation_labels, delimiter=',')


# In[ ]:





# In[ ]:


# clear backend session of tf
# tf.keras.backend.clear_session()


# In[ ]:


print(len(train_features[0]))


# In[ ]:


print(train_labels)


# In[ ]:


# # from numpy import loadtxt
train_features = np.loadtxt('X_train_defective_VGG.csv', delimiter=",")
train_labels = np.loadtxt('y_train_defective_VGG.csv', delimiter=",")
validation_features = np.loadtxt('X_val_defective_VGG.csv', delimiter=",")
validation_labels = np.loadtxt('y_val_defective_VGG.csv', delimiter=",")


# # Training ML Models

# In[ ]:


# Define Evaluation Metric Functions
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def evaluate_model(y_val, y_pred):
    """Function to evaluate model and return the metric of the model

    It returns a dictionary with the classification metrics.
    """
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    result = {"accuracy_score": accuracy,
              "precision_score": precision,
              "recall_score": recall,
              "f1_score": f1}
    return result


def plot_confusion_matrix(y_val, y_pred, label):
    '''function to plot confusion matrix

    Args
    y_val: array. The validation set of the target variable.
    y_pred: array. Model's prediction.
    label: list. A list containing all the classes in the target variable

    Returns
    It returns a plot of the confusion matrix
    '''
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm, display_labels=label).plot(ax=ax, values_format='', xticks_rotation='vertical')
    plt.show()


def display_predictions(y_test, y_pred):
    """
    Display actual values and model predictions in a Pandas DataFrame for the first 10 instances.

    Args:
    y_test: true labels of the test set
    y_pred: model's prediction

    Returns:
    Pandas DataFrame containing actual values and model predictions for the first 10 instances
    """
    df_results = pd.DataFrame({"Actual": y_test[:10],
                               "Prediction": y_pred[:10]})
    return df_results


# class labels
label = ['Defective', 'Good']
RANDOM_STATE = 1


# In[ ]:


def train_models(X_train, X_test, y_train, y_test, model,
                 title, parameters, label=label):
    '''Function to train ML algorithms.'''

    # Train the algorithm
    print("Training the {} algorithm...".format(title))
    # Fine-tune model
    grid_search = GridSearchCV(estimator=model,
                               param_grid=parameters,
                               scoring='f1',
                               cv=5,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # get the parameters that gave the best score
    best_parameters = grid_search.best_params_
    print("Best Parameters:", best_parameters)
    # extract the best model
    best_model = grid_search.best_estimator_

    # Evaluate model on the training set
    print("Training Result - {}".format(title))
    print(evaluate_model(y_train, best_model.predict(X_train)))
    # Evaluate model on the test set
    y_pred = best_model.predict(X_test)
    result = evaluate_model(y_test, y_pred)
    print(result)
    plot_confusion_matrix(y_test, y_pred, label)
    print(classification_report(y_test, y_pred))
    # compare the actual and predicted values
    display(display_predictions(y_test, y_pred))
    print("--------Done---------")


# ### Logistic Regression

# In[ ]:


# Create a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Define different parameter combinations to search over
parameters = {
    'C': [0.0001, 0.001, 0.01],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear']
    }

# Train models
train_models(train_features,
             validation_features,
             train_labels,
             validation_labels,
             label=label,
             model=lr_model,
             parameters=parameters,
             title="Logistic Regression")


# ### Random Forest Model

# In[ ]:


rf_model = RandomForestClassifier()

# setting different parameter combinations
parameters = [{'min_samples_split': np.arange(10,70,10),
               'criterion': ['gini', 'entropy'],
               'n_estimators': np.arange(100,200,20),
               'max_depth': [4, 5, 6]}
              ]

# Use the best model
train_models(train_features,
             validation_features,
             train_labels,
             validation_labels,
             label=label,
             model=rf_model,
             parameters=parameters,
             title="Random Forest")


# ### Support Vector Machine

# In[ ]:


# Create an SVC model
svm_model = svm.SVC()

# Define different parameter combinations to search over
parameters = {
    'C': [0.0001, 0.001, 0.01],  # Regularization parameter
    'kernel': ['linear', 'poly']  # Kernel type
    }

# Train models
train_models(train_features,
             validation_features,
             train_labels,
             validation_labels,
             label=label,
             model=svm_model,
             parameters=parameters,
             title="Support Vector Machine")


# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
accuracy_score_tf = [94.47, 92.39, 90.36, 94.00, 95.01, 97.24]
precision_score_tf = [94.48, 92.71, 90.36, 94.03, 95.02, 97.25]
recall_score_tf = [94.47, 92.39, 90.36, 94.00, 95.01, 97.24]
f1_score_tf = [94.46, 92.33, 90.36, 94.01, 95.01, 97.24]

models = ["ResNet50 + Logistic Regression", "ResNet50 + Random Forest", "ResNet50 + SVM",
          "VGG19 + Logistic Regression", "VGG19 + Random Forest", "VGG19 + SVM"]

# Colors
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.15
X = np.arange(len(models))

for i, metric in enumerate([accuracy_score_tf, precision_score_tf, recall_score_tf, f1_score_tf]):
    ax.bar(X + i * bar_width, metric, color=colors[i], width=bar_width, label=["Accuracy", "Precision", "Recall", "F1"][i])

plt.xticks(X + 1.5 * bar_width, models, rotation=45, ha="right")
plt.xlabel("ML Algorithms")
plt.ylabel("Training Score")
plt.title("Training Result")
# Move the legend to the bottom right
plt.legend(loc='lower right')
plt.show()


# In[ ]:


# Create empty list for accuracy, precision, recall and f1-score of each ml algorithm

accuracy_score_tf = [82.26, 79.03, 79.84, 85.48, 82.80, 85.48]
precision_score_tf = [82.23, 79.45, 79.81, 85.48, 82.78, 85.47]
recall_score_tf = [82.26, 79.03, 79.84, 85.48, 82.80, 85.48]
f1_score_tf = [82.22, 78.70, 79.82, 85.48, 82.75, 85.47]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.15
X = np.arange(len(models))

for i, metric in enumerate([accuracy_score_tf, precision_score_tf, recall_score_tf, f1_score_tf]):
    ax.bar(X + i * bar_width, metric, color=colors[i], width=bar_width, label=["Accuracy", "Precision", "Recall", "F1"][i])

plt.xticks(X + 1.5 * bar_width, models, rotation=45, ha="right")
plt.xlabel("ML Algorithms")
plt.ylabel("Validation Score")
plt.title("Validation Result")
# Move the legend to the bottom right
plt.legend(loc='lower right')
plt.show()


# In[ ]:




