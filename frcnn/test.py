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
from matplotlib import pyplot as plt
import os
from sklearn.metrics import average_precision_score

from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn.Config import Config
from keras_frcnn.utils import get_data,get_real_coordinates
from keras_frcnn.utils import format_img
from keras_frcnn.utils import rpn_to_roi,non_max_suppression_fast,apply_regr, get_map
from optparse import OptionParser


# Configuration 
base_path = ''

test_path =  'annotation.txt' # Training data (annotation file)

# Augmentation flag
horizontal_flips = True # Augment with horizontal flips in training. 
vertical_flips = True   # Augment with vertical flips in training. 
rot_90 = True           # Augment with 90 degree rotations in training. 

output_weight_path = os.path.join(base_path, 'model/model__frcnn__vgg.hdf5')

commands = OptionParser()
commands.add_option("--map",dest="getMap", help="Get mAP of model.", action="store_true",default=False)
commands.add_option("--d",dest="modelPath", help="Provide directory of saved weights.",default=output_weight_path)
(options, args) = commands.parse_args()


record_path = os.path.join(base_path, 'model/record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')

# Create the config
C = Config()
num_rois = C.num_rois # Number of RoIs to process at once.
C.use_horizontal_flips = horizontal_flips
C.use_vertical_flips = vertical_flips
C.rot_90 = rot_90

C.record_path = record_path
C.model_path = options.modelPath
input_size = C.im_size


if C.network == 'mobilenetv1':
    from keras_frcnn import mobilenet as nn
elif C.network == 'mobilenetv2':
    from keras_frcnn import mobilenetv2 as nn
elif C.network == 'resnet':
    from keras_frcnn import resnet as nn
else:
    from keras_frcnn import vgg16 as nn
#-----------------------------------------------------
# This step will spend some time to load the data        #
#--------------------------------------------------------#
st = time.time()
test_imgs, classes_count, class_mapping = get_data(test_path, mode="TEST")
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

input_shape_img = (input_size, input_size, 3)

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


# Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
     
test_base_path = "test"
result_path = "results/"

if not options.getMap:
    test_imgs = os.listdir(test_base_path)

imgs_path = []
    

# If the box classification value is less than this, we ignore this box
bbox_threshold = 0.1
T = {}
P = {}
for idx, img_data in enumerate(test_imgs):
    if options.getMap:
        filename = os.path.basename(img_data['filepath']).split('/')[-1]
        filepath = img_data['filepath']
    else:
        filename = img_data
        filepath = os.path.join(test_base_path,filename)
    if not filename.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    
    st = time.time()
    
    img = cv2.imread(filepath)

    if options.getMap:
        # Add ground truth boxes
        for bboxes in img_data['bboxes']:
            gt_x1,gt_x2 = bboxes['x1'], bboxes['x2']
            gt_y1,gt_y2 = bboxes['y1'], bboxes['y2']
            cv2.rectangle(img,(gt_x1,gt_y1), (gt_x2, gt_y2), (255,255,255),1)

    X,fx,fy = format_img(img, C)
    # c h w
    # 0 h w c
    X = np.transpose(X, (0, 2, 3, 1))
    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2] = model_rpn.predict(X)

    # Get bboxes by applying NMS 
    # R.shape = (300, 4)
    R = rpn_to_roi(Y1, Y2, C,K.set_image_data_format('channels_last'), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    for jk in range(R.shape[0]//C.num_rois + 1):
        if jk >= 16:
            break
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        # print(ROIs);
        if ROIs.shape[1] == 0:
            break
        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
        [P_cls, P_regr] = model_classifier_only.predict([X, ROIs])
        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue
            print(P_cls[0,ii,:])
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    detections = []
    all_dets = []
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)
            # Calculate real coordinates on original image
            # (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            real_x1 = int(round(x1*fx))
            real_x2 = int(round(x2*fx))
            real_y1 = int(round(y1*fy))
            real_y2 = int(round(y2*fy))
            cv2.rectangle(img,(real_x1,real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            detections.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.6,1)
            textOrg = (real_x1,real_y1+24)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX,0.6, (0, 0, 0), 1)
            

    print('Elapsed time = {}'.format(time.time() - st))
    print(filename)
    print(detections)
    cv2.imwrite(result_path + filename, img)
    print("\""+ result_path + filename+ "\"")

    if options.getMap:
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])

if options.getMap:
    print("----------------\nmAP and accuracy from test images\n-------------------")
    T_all = []
    P_all = []
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        all_aps.append(ap)
        T_all.extend(T[key])
        P_all.extend(P[key])
        print('{} AP: {}'.format(key, ap))
        print("\tT:" + str(T[key]))
        print("\tP:" + str(P[key]))
    print('mAP = {}'.format(np.mean(np.array(all_aps))))
    # Calculate Accuracy
    TP = 0 
    TN = 0
    for i in range(len(T_all)):
        if T_all[i] != 0 and P_all[i] != 0:
            TP+=1
        elif T_all[i] == 0 and P_all[i] == 0:
            TN+=1 

print('Accuracy = {}'.format((TP+TN)/float(len(T_all))))

try:
    import winsound
    winsound.Beep(37,100)
    winsound.Beep(300,250)
    winsound.Beep(400,500)
    winsound.Beep(600,500)
except:
    print("")
