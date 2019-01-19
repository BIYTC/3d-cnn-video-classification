from keras.models import Input
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
import keras
import h5py


def cnn_3d(num_classes):
    ## input layer
    input_layer = Input((100, 150, 150, 3))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)

    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=num_classes, activation='softmax')(dense_layer2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    return model