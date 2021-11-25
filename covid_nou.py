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

fig = pyplot.figure(figsize=(32,32))
for i in range(x_train.shape[0]):
    ax = fig.add_subplot(int(x_train.shape[0]/2), int(x_train.shape[0]/2), i+1)
    ax.title.set_text(labels[y_train[i]])
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

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

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=config['train']['lr']), metrics=['accuracy'])
callbacks =[keras.callbacks.CSVLogger("file.csv{local_dt}", separator="," , append=False)]
NUM_EPOCHS = config['train']['n_epochs']
history = model.fit(train_batches, steps_per_epoch = len(train_batches) ,validation_data = validation_batches, validation_steps = len(validation_batches), epochs= NUM_EPOCHS , callbacks=callbacks)

model.save('damn.h5')  

plot_acc_loss(history)
test_generator = validation_datagen.flow_from_directory(dataset_dir + '/test', target_size=config['net']['img'], batch_size=config['train']['bs'], class_mode='binary')
x_test,y_test=test_generator[0]
print(x_test.shape,y_test.shape)
probs=model.predict_generator(test_generator,14)
print(probs)


confusion_matrix_manual=conf_mat(probs,y_test)
preds=convert_prob(probs)
cm=confusion_matrix(preds,y_test)
print("Matrice de confuzie calculata manual : ", confusion_matrix_manual,"si caculata cu functia sklearn:",cm)

accuracy,senzitivity,specifity,precision,recall,f1=metrics(confusion_matrix_manual,y_test)
print("ACC:",accuracy,"TPR:",senzitivity,"TNR:",specifity,"PPV:" ,precision,"FPR:",recall,"f1:",f1)

acc=accuracy_score(preds,y_test)
preci=precision_score(preds,y_test)
reca=recall_score(preds,y_test)
F1=f1_score(preds,y_test)
print("acc",acc,"PPV",preci,"FPR",reca,"f1:",F1)

ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr,tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
print('test acc:', test_acc)