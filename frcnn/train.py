#  Libraries 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# rom __future__ import division
# from __future__ import print_function
# from __future__ import absolute_import
import random
import pprint
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
from keras.losses import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras_frcnn.Config import Config
from keras_frcnn.utils import get_data,get_anchor_gt
from keras_frcnn.utils import rpn_to_roi, calc_iou
from keras_frcnn.utils import class_loss_cls,class_loss_regr,rpn_loss_cls,rpn_loss_regr
from optparse import OptionParser

commands = OptionParser()
commands.add_option("--v",dest="viewDebug", help="View debug image.",action="store_true",default=False)
commands.add_option("--c",dest="previousWeight", help="Continue from specified weights.",default=None)
commands.add_option("--r",dest="restart", help="Restart training from pretrained weights.", action="store_true",default=False)
commands.add_option("--E",dest="rpn_err", help="Check if there are errors on rpn generation.",action="store_true",default=False)
commands.add_option("--t",dest="testing", help="Proceed to testing.",action="store_true",default=False)
(options, args) = commands.parse_args()

# Training 

base_path = ''

train_path =  'annotation.txt' # Training data (annotation file)

# Augmentation flag
horizontal_flips = True # Augment with horizontal flips in training. 
vertical_flips = True   # Augment with vertical flips in training. 
rot_90 = True           # Augment with 90 degree rotations in training. 

output_weight_path = os.path.join(base_path, 'model/model__frcnn__vgg.hdf5')

record_path = os.path.join(base_path, 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')

# Create the config
C = Config()

C.use_horizontal_flips = horizontal_flips
C.use_vertical_flips = vertical_flips
C.rot_90 = rot_90

C.record_path = record_path
C.model_path = output_weight_path
if options.previousWeight != None:
    C.model_path = options.previousWeight
num_rois = C.num_rois # Number of RoIs to process at once.

input_size = [C.im_size, C.im_size]

if C.network == 'mobilenetv1':
    print("Model: " + C.network)
    from keras_frcnn import mobilenet as nn
elif C.network == 'mobilenetv2':
    print("Model: " + C.network)
    from keras_frcnn import mobilenetv2 as nn
elif C.network == 'resnet':
    print("Model: " + C.network)
    from keras_frcnn import resnet as nn
else:
    print("Model: " + C.network)
    from keras_frcnn import vgg16 as nn
     
base_weight_path = os.path.join(base_path, nn.get_weight_path())
C.base_net_weights = base_weight_path
#-----------------------------------------------------
# This step will spend some time to load the data        #
#--------------------------------------------------------#
st = time.time()
train_imgs, classes_count, class_mapping = get_data(train_path)
#val_imgs, classes_count, class_mapping = get_data(train_path, mode = "VAL")
print()
print('Spend %0.2f mins to load the data' % ((time.time()-st)/60) )

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)
# e.g.
#    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
#    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
C.class_mapping = class_mapping

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print(class_mapping)
# Save the configuration
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

# Shuffle the images with seed
random.seed(1)
random.shuffle(train_imgs)

print('Num train samples (images) {}'.format(len(train_imgs)))


# Get train data generator which generate X, Y, image_data
data_gen_train = get_anchor_gt(train_imgs, C, nn.get_img_output_length, mode='train')
# data_gen_val = get_anchor_gt(val_imgs, C, nn.get_img_output_length, mode='val')

if options.rpn_err:
    errorInAnchors = 0
    for i in range(len(train_imgs)):
        X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
        if image_data['bboxes'][0]['class'] != 'bg' and debug_num_pos <= 0:
            errorInAnchors+=1
        
    if errorInAnchors > 0:
        print("There are errors in generating anchors: "+ str(errorInAnchors))
        exit()

class_list = {v: k for k, v in class_mapping.items()}
while True:
    X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
    if C.classView == -1:
        C.viewPos -= 1
        if C.viewPos <= 0:
             break
        continue
    if image_data['bboxes'][0]['class'] == class_list[C.classView]:
        C.viewPos -= 1
    if(C.viewPos <= 0):
        break
# print('Original image: height=%d width=%d'%(image_data['height'], image_data['width']))
# print('Resized image:  height=%d width=%d C.im_size=%d'%(X.shape[1], X.shape[2], C.im_size))
# print('Feature map size: height=%d width=%d C.rpn_stride=%d'%(Y[0].shape[1], Y[0].shape[2], C.rpn_stride))
# print(X.shape)
# print(str(len(Y))+" includes 'y_rpn_cls' and 'y_rpn_regr'")
# print('Shape of y_rpn_cls {}'.format(Y[0].shape))
# print('Shape of y_rpn_regr {}'.format(Y[1].shape))
print(image_data)

print('Number of positive anchors for this image: %d' % (debug_num_pos))

img = debug_img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# gt_x1, gt_x2 = image_data['bboxes'][0]['x1']*(X.shape[2]/image_data['height']), image_data['bboxes'][0]['x2']*(X.shape[2]/image_data['height'])
# gt_y1, gt_y2 = image_data['bboxes'][0]['y1']*(X.shape[1]/image_data['width']), image_data['bboxes'][0]['y2']*(X.shape[1]/image_data['width'])
# gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

# color = (0, 255, 0)
# cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
# cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

cls = Y[0][0]
pos_cls = np.where(cls==1)
# print(pos_cls)
regr = Y[1][0]
pos_regr = np.where(regr==1)
# print(pos_regr)
# print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0],pos_cls[1][0],:]))
# print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0],pos_regr[1][0],:]))

img = debug_img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
color = (0, 255, 0)

for i,bboxes in enumerate(image_data['bboxes']):
    gt_x1, gt_x2 = image_data['bboxes'][i]['x1']*(X.shape[2]/image_data['width']), image_data['bboxes'][i]['x2']*(X.shape[2]/image_data['width'])
    gt_y1, gt_y2 = image_data['bboxes'][i]['y1']*(X.shape[1]/image_data['height']), image_data['bboxes'][i]['y2']*(X.shape[1]/image_data['height'])
    gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

    #   cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
    cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

    # Add text
    textLabel = "gt"
    (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
    textOrg = (gt_x1, gt_y1+5)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
    cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
    # cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

# Draw positive anchors according to the y_rpn_regr
for i in range(debug_num_pos):

    color = (100+i*(155/4), 0, 100+i*(155/4))

    idx = pos_regr[2][i*4]/4
    anchor_size = C.anchor_box_scales[int(idx/4)]
    anchor_ratio = C.anchor_box_ratios[2-int((idx+1)%4)]

    center = (pos_regr[1][i*4]*C.rpn_stride, pos_regr[0][i*4]*C.rpn_stride)
    print('Center position of positive anchor: ', center)
    cv2.circle(img, center, 3, color, -1)
    anc_w, anc_h = anchor_size*anchor_ratio[0], anchor_size*anchor_ratio[1]
    cv2.rectangle(img, (center[0]-int(anc_w/2), center[1]-int(anc_h/2)), (center[0]+int(anc_w/2), center[1]+int(anc_h/2)), color, 2)
# #         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


if options.viewDebug:
    print('Green bboxes is ground-truth bbox. Others are positive anchors')
    plt.figure(figsize=(8,8))
    plt.grid()
    plt.imshow(img)
    plt.show()
    exit()

# Build Model
input_shape_img = (input_size[0], input_size[1], 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))


# define the base network (VGG here, can be Resnet50, Inception, etc)
# shared_layers = nn_base(img_input, trainable=True,compressor =compression)
shared_layers = nn.nn_base(img_input, trainable=True)
     
# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
# rpn = rpn_layer(shared_layers, num_anchors,compressor=compression)
# classifier = classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count),compressor =compression)

rpn = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# Because the google colab can only run the session several hours one time (then you need to connect again), 
# we need to save the model and load the model to continue training
if not os.path.isfile(C.model_path) or options.restart:
    #If this is the begin of the training, load the pre-traind base network such as vgg-16
    try:
        print('Fresh train')
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')
    
    # Create the record.csv file to record losses, acc and mAP
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
else:
    # If this is a continued training, load the trained model from before
    print('Continue training based on previous trained model')
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    if options.previousWeight != None:
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
    else:
        # Load the records
        record_df = pd.read_csv(record_path)

        r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
        r_class_acc = record_df['class_acc']
        r_loss_rpn_cls = record_df['loss_rpn_cls']
        r_loss_rpn_regr = record_df['loss_rpn_regr']
        r_loss_class_cls = record_df['loss_class_cls']
        r_loss_class_regr = record_df['loss_class_regr']
        r_curr_loss = record_df['curr_loss']
        r_elapsed_time = record_df['elapsed_time']
        r_mAP = record_df['mAP']

        print('Already trained %d epochs'% (len(record_df)))

# optimizer = Adam(lr=1e-5)
# optimizer_classifier = Adam(lr=1e-5)

# optimizer = Adam(lr=1e-3, clipnorm=0.001)
# optimizer_classifier = Adam(lr=1e-3, clipnorm=0.001)
if C.network == 'resnet':
    # print("Optimizer: SSD")
    # optimizer = SGD(learning_rate=C.epsilon, decay=0.0005, momentum=0.9)
    # optimizer_classifier = SGD(learning_rate=C.epsilon, decay=0.0005, momentum=0.9)
    rpnLR = C.epsilon
    clsLR = C.epsilon * C.multiplier
    print("RPN Optimizer: ADAM - " + str(rpnLR))
    optimizer = Adam(learning_rate=rpnLR, clipnorm=0.001)
    print("Classifier Optimizer: ADAM - " + str(clsLR))
    optimizer_classifier = Adam(learning_rate=clsLR, clipnorm=0.001)
elif C.network == 'mobilenetv1' or C.network == 'mobilenetv2':
    rpnLR = C.epsilon
    clsLR = C.epsilon * C.multiplier
    print("RPN Optimizer: ADAM - " + str(rpnLR))
    optimizer = Adam(learning_rate=rpnLR, clipnorm=0.001)
    print("Classifier Optimizer: ADAM - " + str(clsLR))
    optimizer_classifier = Adam(learning_rate=clsLR, clipnorm=0.001)
    # print("Optimizer: SSD")
    # optimizer = SGD(learning_rate=rpnLR, decay=0.0005, momentum=0.9)
    # optimizer_classifier = SGD(learning_rate=clsLR, decay=0.0005, momentum=0.9)
    # print("Optimizer: SSD - " + C.network )
    # optimizer = SGD(learning_rate=C.epsilon, decay=0.0005, momentum=0.9)
    # optimizer_classifier = SGD(learning_rate=C.epsilon, decay=0.0005, momentum=0.9)
else:
    print("RPN Optimizer: ADAM - " + str(C.epsilon))
    optimizer = Adam(learning_rate=C.epsilon, clipnorm=0.001)
    print("Classifier Optimizer: ADAM - " + str(C.epsilon-(C.epsilon/4)))
    optimizer_classifier = Adam(learning_rate=C.epsilon-(C.epsilon/4), clipnorm=0.001)

model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors, C), rpn_loss_regr(num_anchors, C)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1 , C)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# Training setting
total_epochs = len(record_df)
r_epochs = len(record_df)
num_epochs = C.epochs
iter_num = 0
epoch_length = 500
total_epochs += num_epochs
losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

if len(record_df)==0:
    best_loss = np.Inf
else:
    best_loss = np.min(r_curr_loss)

print("Best Loss : " + str(best_loss))
prev_best = best_loss
start_time = time.time()

for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
    
    r_epochs += 1

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
#                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
            X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)
            # val_X, val_Y, val_img_data, val_debug_img, val_debug_num_pos = next(data_gen_val)

            # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
            loss_rpn = model_rpn.train_on_batch(X, Y)
            #val_loss_rpn = model_rpn.test_on_batch(val_X,val_Y)


            # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
            P_rpn = model_rpn.predict_on_batch(X)
            # val_P_rpn = model_rpn.predict_on_batch(val_X)

            # R: bboxes (shape=(300,4))
            # Convert rpn layer to roi bboxes
            R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.set_image_data_format('channels_last'), use_regr=True, overlap_thresh=0.4, max_boxes=300)
            # val_R = rpn_to_roi(val_P_rpn[0], val_P_rpn[1], C, K.set_image_data_format('channels_last'), use_regr=True, overlap_thresh=0.4, max_boxes=300)
           
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
            # Y1: one hot code for bboxes from above => x_roi (X)
            # Y2: corresponding labels and corresponding gt bboxes
            X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)
            # val_X2, val_Y1, val_Y2, val_IouS = calc_iou(val_R, val_img_data, C, class_mapping)

            # If X2 is None means there are no matching bboxes
            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            
            # Find out the positive anchors and negative anchors
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                
                # Randomly choose (num_rois - num_pos) neg samples
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                
                # Save all the pos and neg samples in sel_samples
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            # training_data: [X, X2[:, sel_samples, :]]
            # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
            #  X                     => img_data resized image
            #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
            #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
            #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                      ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))
                    elapsed_time = (time.time()-start_time)/60

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(output_weight_path)
                    try:
                        import winsound
                        winsound.Beep(37,100)
                        winsound.Beep(1500,250)
                    except:
                        print("")
                new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                           'class_acc':round(class_acc, 3), 
                           'loss_rpn_cls':round(loss_rpn_cls, 3), 
                           'loss_rpn_regr':round(loss_rpn_regr, 3), 
                           'loss_class_cls':round(loss_class_cls, 3), 
                           'loss_class_regr':round(loss_class_regr, 3), 
                           'curr_loss':round(curr_loss, 3), 
                           'elapsed_time':round(elapsed_time, 3), 
                           'mAP': 0}
                record_df = record_df.append(new_row, ignore_index=True)
                record_df.to_csv(record_path, index=0)
               
                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
plt.title('mean_overlapping_bboxes')
plt.subplot(1,2,2)
plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
plt.title('class_acc')

plt.savefig('accuracyplot.png')
# plt.show()

# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
# plt.title('loss_rpn_cls')
# plt.subplot(1,2,2)
# plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
# plt.title('loss_rpn_regr')
# plt.show()


# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
# plt.title('loss_class_cls')
# plt.subplot(1,2,2)
# plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
# plt.title('loss_class_regr')
# plt.show()
# plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
# plt.title('total_loss')
# plt.show()

# #################################

# plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
# plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
# plt.title('total_loss')
# plt.subplot(1,2,2)
# plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
# plt.title('elapsed_time')
# plt.show()
    
plt.figure(figsize=(15,5))
plt.title('Loss(M): rpn_cls(B), rpn_regr(G), class_cls(R), cls_regr(C)')
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'b')
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'g')
plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'c')
plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'm')
plt.savefig("lossesplot.png")

if best_loss < prev_best:
    print("Previous Best: "+ str(prev_best))
    print("New Best: "+ str(best_loss))

try:
    # playsund==1.2.2
    import winsound
    winsound.Beep(37,100)
    winsound.Beep(300,250)
    winsound.Beep(400,500)
    winsound.Beep(600,500)
except:
    print("")

if options.testing and best_loss < prev_best:
    os.system("python test.py --map")
elif best_loss >= prev_best:
    print("No better than previous best loss.")

# plt.show()
print('Training complete, exiting.')
