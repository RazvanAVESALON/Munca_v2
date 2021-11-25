
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy import unique
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plot_acc_loss import plot_acc_loss 
from tensorflow.keras.applications import VGG16
import yaml

config = None
with open('config.yml') as f: # reads .yml/.yaml files
    config = yaml.load(f)

dataset_dir  = config['net']['dir']
BATCH_SIZE = config['train']['bs']

validation_datagen = ImageDataGenerator(rescale=1./255)    

new_model = tf.keras.models.load_model('damn.h5')
new_model.summary()

test_generator = validation_datagen.flow_from_directory(dataset_dir + '/test', target_size=config['net']['img'], batch_size=config['train']['bs'], class_mode='binary')
test_loss, test_acc = new_model.evaluate_generator(test_generator, steps=len(test_generator))
print('test acc:', test_acc)