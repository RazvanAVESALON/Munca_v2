
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy import load, unique
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plot_acc_loss import plot_acc_loss 
from tensorflow.keras.applications import VGG16
import yaml
from confusion_matrix_metrics import convert_prob , conf_mat, metrics
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix, accuracy_score,recall_score,f1_score,precision_score,plot_confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd 
config = None
with open('config.yml') as f: # reads .yml/.yaml files
    config = yaml.load(f)

dataset_dir  = config['net']['dir']
BATCH_SIZE = config['train']['bs']

validation_datagen = ImageDataGenerator(rescale=1./255)    
network=pd.read_csv (r"D:\ai intro\Munca_v2\file1911_12022021.csv")
new_model = tf.convert_to_tensor(network)
new_model.summary()

test_generator = validation_datagen.flow_from_directory(dataset_dir + '/test', target_size=config['net']['img'], batch_size=config['train']['bs'], class_mode='binary')
x_test,y_test=test_generator[0]
print(x_test.shape,y_test.shape)
probs=new_model.predict_generator(test_generator,1)

print(y_test.shape,probs.shape) 

confusion_matrix_manual=conf_mat(probs,y_test)
preds=convert_prob(probs)
cm=confusion_matrix(preds,y_test)
print("Matrice de confuzie calculata manual : ", confusion_matrix_manual,"si caculata cu functia sklearn:",cm)

accuracy,senzitivity,specifity,precision,FPR,f1=metrics(confusion_matrix_manual,y_test)
print("ACC:",accuracy,"TPR:",senzitivity,"TNR:",specifity,"PPV:" ,precision,"FPR:",FPR,"f1:",f1)

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


test_loss, test_acc = new_model.evaluate_generator(test_generator, steps=len(test_generator))
print('test acc:', test_acc)