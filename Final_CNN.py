# -*- coding: utf-8 -*-
"""
Created on Thu May  4 02:04:38 2023

@author: reena
"""

# Import the libraries
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

# Define the path to read the images
crack = Path('C:/Users/reena/OneDrive - Lamar University/Desktop/Machine learning/CNN/Crack')
Nocrack = Path('C:/Users/reena/OneDrive - Lamar University/Desktop/Machine learning/CNN/Nocrack')

# Function to create a dataframe with filepath and label
def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df

# Create Dataframes for crack and nocrack images
crack_df = generate_df(crack, label="CRACK")
nocrack_df = generate_df(Nocrack, label="NOCRACK")

# Concatenate the two dataframes
all_df = pd.concat([crack_df, nocrack_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
all_df

# Split the data to traint and test set
train_df, test_df = train_test_split(all_df,train_size=0.7,shuffle=True,random_state=1)

# Preprocess the image by rescaling the pixel values
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Generate train and test datasets
# Resize the image to 120 X 120
# Get the image in RGB format
# Set the number of samples for each batch
train_data = train_gen.flow_from_dataframe(train_df,x_col='Filepath',
    y_col='Label',target_size=(120, 120),color_mode='rgb',
    class_mode='binary',batch_size=32,shuffle=True,seed=42,subset='training')

test_data = train_gen.flow_from_dataframe(test_df,x_col='Filepath',y_col='Label',
    target_size=(120, 120),color_mode='rgb',class_mode='binary',
    batch_size=32,shuffle=False,seed=42)

# Building the model
# Create an input layer of model with specified shape
inputs = tf.keras.Input(shape=(120, 120, 3))
# Add a 2D convolutional layer to the model with 16 filters
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
# Add max pooling layer to the model
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
# Add a 2D convolutional layer to the model with 32 filters
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
# Add Global average pooling layer to model
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Add dense layer to model with 1 neuron
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model using adam as optimizer
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Get the summary of the model architecture
print(model.summary())

# Train the model for 10 epochs
history = model.fit(train_data,epochs=10)

# Evaluate the trained model performance on test data
def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
evaluate_model(model, test_data)

# Get the loss and accuracy of model on test data
score=model.evaluate(test_data)
print('Final Accuracy : ', score[1]*100, "%" )
print('Final Loss : ', score[0])

# Load the image
img = cv2.imread('C:/Users/reena/OneDrive - Lamar University/Desktop/Machine learning/CNN/Test/1.jpg')

# Resize the image to the input size of the model
img = cv2.resize(img, (120, 120))

# Rescale the pixel values to be between 0 and 1
img = img / 255.0

# Make a prediction using the trained model
prediction = model.predict(img.reshape(1, 120, 120, 3))[0]

# Print the prediction (0 = no crack, 1 = crack)
if prediction < 0.5:
    print("No crack detected")
else:
    print("Crack detected")
    
# Get the predicted probabilities for the test data
y_pred = model.predict(test_data).ravel()

# Compute the false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(test_data.classes, y_pred)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random guess')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()