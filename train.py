import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from dataset import *
from model import cnn_3d
import argparse
from keras.optimizers import Adadelta
from utils import*

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='path to dataset')
opt = parser.parse_args()

# image dimensions
video_length=50
img_h=50
img_w=50
img_c=3

# Parameters
params = {'dim': (video_length,img_h,img_w),
          'batch_size': 2,
          'n_classes': 2,
          'n_channels': img_c,
          'shuffle': True}

# Datasets
partition, labels = load_data(opt.data)

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['val'], labels, **params)

# Design model
model = cnn_3d(num_classes=params['n_classes'])
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=0.1), metrics=['acc'])

#checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
checkpoint = ModelCheckpoint(filepath='./snapshots/action.h5', monitor='loss', verbose=1, mode='min', period=1)
# Train model on dataset
model.fit_generator(generator=training_generator,
                    #steps_per_epoch=int(training_generator.n / params['batch_size']),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=validation_generator,
                    #validation_steps=int(validation_generator.n / params['batch_size']))
)