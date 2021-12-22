import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import tensorflow_datasets as tfds
from numpy import unique
from numpy import argmax
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plot_acc_loss import plot_acc_loss 
from tensorflow.keras.applications import VGG16
import yaml
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix, accuracy_score,recall_score,f1_score,precision_score,plot_confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from confusion_matrix_metrics import convert_prob , conf_mat, metrics
from datetime import datetime


local_dt=datetime.now()

config = None
with open('config.yml') as f:
    config = yaml.load(f)

dataset_dir  = config['net']['dir']
BATCH_SIZE = config['train']['bs']

train_datagen = ImageDataGenerator(**config['augumentare'])
validation_datagen = ImageDataGenerator(rescale=1./255)     

train_batches = train_datagen.flow_from_directory(dataset_dir + '/train',
                                                  target_size=config['net']['img'],
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode="binary")

validation_batches = validation_datagen.flow_from_directory(dataset_dir + '/val',
                                                  target_size=config['net']['img'],
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode="binary")

x_train, y_train = next(train_batches)
print(x_train.shape)
print(y_train.shape)

labels = {0: 'COVID', 1: 'Normal'}

fig = plt.figure(figsize=(32,32))
for i in range(x_train.shape[0]):
    ax = fig.add_subplot(int(x_train.shape[0]/2), int(x_train.shape[0]/2), i+1)
    ax.title.set_text(labels[y_train[i]])
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

in_shape = (config['net']['img'][0], config['net']['img'][1], 3)

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=in_shape)
conv_base.summary()
n1=config['net']['n1']

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(n1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
conv_base.trainable = False

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=config['train']['lr']), metrics=['accuracy'])
callbacks =[keras.callbacks.CSVLogger(f"file{datetime.now().strftime('%H%M_%m%d%Y')}.csv", separator="," , append=False)]
NUM_EPOCHS = config['train']['n_epochs']
history = model.fit(train_batches, steps_per_epoch = len(train_batches) ,validation_data = validation_batches, validation_steps = len(validation_batches), epochs= NUM_EPOCHS , callbacks=callbacks)

model.save(f"damn{datetime.now().strftime('%H%M_%m%d%Y')}.h5")  

plot_acc_loss(history)
