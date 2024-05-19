import numpy as np
import math 
import numpy

from csv import reader
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Activation, Dropout, Dense

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

from keras.utils import to_categorical

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing as pre
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ResNet only accuracy: ==> %
ResNet = pd.read_csv("ResNet50Features.csv", header=None)
ResNet_arr = np.array(ResNet)
ResNet_arr = np.nan_to_num(ResNet_arr)

ResNet_norm = pre.MinMaxScaler().fit_transform(ResNet_arr)
ResNet_norm = ResNet_norm[:, 0:1000]

# VGG only accuracy: ==> %
VGG_16 = pd.read_csv("VGG16Features.csv", header=None)
VGG_16_arr = np.array(VGG_16)
VGG_16_arr = np.nan_to_num(VGG_16_arr)

VGG_16_norm = pre.MinMaxScaler().fit_transform(VGG_16_arr)
VGG_16_norm = VGG_16_norm[:, 0:1000]

X_train = np.array(ResNet_norm)
# X_train = np.array(VGG_16_norm)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
label_train = pd.read_csv("label.csv",header=None)
label_train = np.array(label_train)

n_classes=50
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(1000, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu')) # 91.6%
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu')) # 91% -- 0.35 || 93.7% -- 0.2 || 69.9% -- 0.5 || 91.6% -- 0.3
model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=256, kernel_size=2, activation='relu')) #  -- 0.2 ||  -- 0.1
# model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
(trainX, testX, trainY, testY) = train_test_split(X_train, label_train, test_size=0.3, random_state=42)
result = model.fit(trainX, trainY, epochs=1000, verbose=1, validation_data=(testX, testY))

# Plotting Accuracy
plt.plot(result.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.savefig("Accuracy_Res_over_time.png")
# plt.savefig("Accuracy_VGG_over_time.png")

plt.clf()

# Plotting Loss
plt.plot(result.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')
plt.savefig("Loss_Res_over_time.png")
# plt.savefig("Loss_VGG_over_time.png")

_, accuracy = model.evaluate(testX, testY, verbose=0)
print("accuracy:")
print(accuracy)

pred = model.predict(testX)
pred_y = pred.argmax(axis=-1)
cm = confusion_matrix(testY, pred_y)
print(cm)

# Get class labels
labels = [str(i) for i in range(cm.shape[0])]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig("ResNet_Confusion.png")
# plt.savefig("VGG_Confusion.png")

print(model.summary())