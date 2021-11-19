import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import tensorflow_datasets as tfds
from numpy import unique
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plot_acc_loss import plot_acc_loss 
from tensorflow.keras.applications import VGG16
import yaml
from matplotlib import pyplot

config = None
with open('config.yml') as f: # reads .yml/.yaml files
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

x_test, y_test = next(train_batches)
print(x_test.shape)
print(y_test)

labels = {0: 'COVID', 1: 'Normal'}

fig = pyplot.figure(figsize=(32,32))
for i in range(x_test.shape[0]):
    ax = fig.add_subplot(int(x_test.shape[0]/2), int(x_test.shape[0]/2), i+1)
    ax.title.set_text(labels[y_test[i]])
    pyplot.imshow(x_test[i], cmap=pyplot.get_cmap('gray'))
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
test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
print('test acc:', test_acc)