from tensorflow.keras.models import load_model


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# rom __future__ import division
# from __future__ import print_function
# from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from tensorflow.keras.utils import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.losses import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from tensorflow.keras.layers import Layer, InputSpec
from keras import initializers, regularizers
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.Config import Config
from keras_frcnn.utils import get_data
from optparse import OptionParser


commands = OptionParser()
commands.add_option("--deploy",dest="deploy", help="Deploy tflite to flutter ap.", action="store_true",default=False)
(options, args) = commands.parse_args()

train_path = 'annotation.txt'
train_imgs, classes_count, class_mapping = get_data(train_path)
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)
C = Config()
if C.network == 'mobilenetv1':
    from keras_frcnn import mobilenet as nn
elif C.network == 'mobilenetv2':
    from keras_frcnn import mobilenetv2 as nn
elif C.network == 'resnet':
    from keras_frcnn import resnet as nn
else:
    from keras_frcnn import vgg16 as nn
C.model_path = "model/model__frcnn__vgg.hdf5"
input_shape_img = (C.im_size, C.im_size, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn.nn_base(img_input)
# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

rpn = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))
model_rpn = Model(img_input, rpn[:2])
model_classifier_only = Model([img_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier_only.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier_only.compile(optimizer='sgd', loss='mse')
currentPath = ""

if options.deploy:
    currentPath = os.getcwd();
    currentPath = os.path.dirname(currentPath)
    currentPath += "\\assets\\"

converter = tf.lite.TFLiteConverter.from_keras_model( model_classifier_only) 
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
model = converter.convert()
file = open( currentPath + 'model_classifier.tflite' , 'wb' ) 
file.write( model )
file.close()

converter = tf.lite.TFLiteConverter.from_keras_model( model_rpn ) 
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
model = converter.convert()
file = open( currentPath + 'model_rpn.tflite' , 'wb' ) 
file.write( model )
file.close()

try:
    # playsund==1.2.2
    import winsound
    winsound.Beep(37,100)
    winsound.Beep(300,250)
    winsound.Beep(400,500)
    winsound.Beep(600,500)
except:
    print("")
print("Done.")
