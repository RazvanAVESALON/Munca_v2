
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from numpy import load, unique
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plot_acc_loss import plot_acc_loss 
from tensorflow.keras.applications import VGG16
import yaml
from confusion_matrix_metrics import convert_prob , conf_mat, metrics
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix, accuracy_score,recall_score,f1_score,precision_score,plot_confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split

config = None
with open('config.yml') as f: # reads .yml/.yaml files
    config = yaml.load(f)

dataset_dir  = config['net']['dir']
BATCH_SIZE = config['train']['bs']

validation_datagen = ImageDataGenerator(rescale=1./255)    
new_model = keras.models.load_model(r"D:\ai intro\Munca_v2\Experimente Confusion Matrix\big dataset\damn2243_01042022.h5")
new_model.summary()

test_generator = validation_datagen.flow_from_directory(dataset_dir + '/test', 
                                                        target_size=config['net']['img'], 
                                                        batch_size=config['train']['bs'], 
                                                        class_mode='binary',
                                                        shuffle=False)
print (test_generator)
probs=new_model.predict_generator(test_generator)

#print(y_test.shape,probs.shape) 

confusion_matrix_manual=conf_mat(probs,test_generator.classes)
preds=convert_prob(probs)
cm=confusion_matrix(preds,test_generator.classes)
print("Matrice de confuzie calculata manual : ", confusion_matrix_manual,"si caculata cu functia sklearn:",cm)

#accuracy,senzitivity,specifity,precision,FPR,f1=metrics(confusion_matrix_manual,test_generator.classes)
#print("ACC:",accuracy,"TPR:",senzitivity,"TNR:",specifity,"PPV:" ,precision,"FPR:",FPR,"f1:",f1)

acc=accuracy_score(preds,test_generator.classes)
preci=precision_score(preds,test_generator.classes)
reca=recall_score(preds,test_generator.classes)
F1=f1_score(preds,test_generator.classes)
print("acc",acc,"PPV",preci,"FPR",reca,"f1:",F1)

ConfusionMatrixDisplay.from_predictions(test_generator.classes, preds)
plt.show()

print(test_generator.classes[:5], type(test_generator.classes))
print(probs[:5])

fpr, tpr, thresholds = roc_curve(test_generator.classes, probs, pos_label=1)
plt.plot(fpr,tpr, marker='.', label='ROC_CURVE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

auc = roc_auc_score(test_generator.classes,  probs)
print('AUC: %.3f' % auc)

precision, recall, thresholds = precision_recall_curve(test_generator.classes, probs)
plt.plot(precision,recall, marker='.', label='Precision_recall')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend()
plt.show()


test_loss, test_acc = new_model.evaluate_generator(test_generator, steps=len(test_generator))
print('test acc:', test_acc)