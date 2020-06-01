# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:51:19 2020

@author: Szedlák Barnabás
"""

#------------------------------------------------------------------------------
# 0. STEP
# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from tensorflow.keras.optimizers import Adam

import segmentation_models
segmentation_models.set_framework('tf.keras') # manually setting tensorflow.keras framework to avoid package collusion

from segmentation_models.losses import JaccardLoss # other losses may be imported here, as well
from segmentation_models.utils import set_trainable
from segmentation_models import Unet
from segmentation_models.metrics import IOUScore # other metrics may be imported here, as well

from vectorizer import train_test_generator # importing functions from our own scipts
import model_backroom as backroom  # importing functions from our own scipts



#------------------------------------------------------------------------------
# 1. STEP
# Initialising vector parameters


SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/"


train_proportion_dict = {"left": 1, #1
                         "midnorth": 8, #15
                         "midcity":33, #27
                         "midsouth":10, #15
                         "right": 45, #36
                         "island": 3} #6


test_file_list = ['right'] # currently all test images are considered with 100% proportion

train_size = 100
train_neighbor_no = 10

test_size = 400
test_neighbor_no = 10

mask_label_dict = {2: [[9,255,247],'Road(2)'],
                   3: [[255,0,4],'Building(3)'],
                   4: [[0,12,255],'Water(4)'],
                   5: [[251,255,0],'Rail(5)'],
                   7: [[120,125,129],'Paved(7)'],
                   8: [[168,124,2],'Rock(8)'],
                   9: [[240,185,32],'Gravel(9)'],
                   10:[[0,134,17],'Forest(10)'],
                   11:[[19,221,36],'Other vegetation(11)'],
                   17:[[0,134,17],'Forest(17)'],
                   18:[[19,221,36],'Other vegetation(18)']}


#------------------------------------------------------------------------------
# 2. STEP
# Building train and test vectors

# manially load normalization values from previously created train sets (if only test set is needed)
pre_norm = np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/vector_norms/" + "minXmaxX_size_20000_04-30-2020-1127.npy" )
pre_norm = None # one of these lines should be commented out

# random vector generator
train_return_dict, test_return_dict = train_test_generator(train_size, 
                                                           train_neighbor_no, 
                                                           train_proportion_dict, 
                                                           test_size, 
                                                           test_neighbor_no,
                                                           test_file_list, 
                                                           pre_norm = pre_norm)


# TRAIN VECTORS
# unpacking train vectors from train return dictionaries
X_train = train_return_dict['X_train'].copy()
y_train = train_return_dict['y_train'].copy()
train_vector_tile_dict = train_return_dict['train_vector_tile_dict'].copy()
train_label_dist_dict = train_return_dict['train_label_dist_dict'].copy()
train_norm_minX_maxX =  train_return_dict['train_norm_minX_maxX'].copy()

# visualising label distribution 
backroom.distribution_inspector(train_label_dist_dict, mask_label_dict, train_vector_tile_dict)

# multidim train vectors
y_train_multidim = backroom.mask_to_multidim(y_train, mask_labels = mask_label_dict.keys())


# TEST VECTORS
# unpacking test vectors from test return dictionaries
X_test = test_return_dict['X_test'].copy()
y_test = test_return_dict['y_test'].copy()
test_vector_tile_dict =  test_return_dict['test_vector_tile_dict'].copy()
test_label_dist_dict = test_return_dict['test_label_dist_dict'].copy()


######### THIS MAY BE USED FOR MULTIPLE TEST VECTORS
# unpacking test vectors from test return dictionaries
test_file = 'right'

X_test = test_return_dict[test_file + '_X_test'].copy()
y_test = test_return_dict[test_file + '_y_test'].copy()
test_vector_tile_dict =  test_return_dict[test_file + '_test_vector_tile_dict'].copy()
test_label_dist_dict = test_return_dict[test_file + '_test_label_dist_dict'].copy()
#########################################################


# visualising label distribution 
backroom.distribution_inspector(test_label_dist_dict, mask_label_dict, test_vector_tile_dict)

# multidim test vectors
y_test_multidim = backroom.mask_to_multidim(y_test, mask_label_dict.keys())



# manual load for verification
# train
X_train = np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/" + "X_train_size_15000_04-05-2020-1146.npy" )
y_train = np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/" + "y_train_size_15000_04-05-2020-1146.npy" )

# test
X_test = np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/" + "X_test_size_3748_04-05-2020-1146.npy" )
y_test = np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/" + "y_test_size_3748_04-05-2020-1146.npy" )
with open("c:/Users/Szedlák Barnabás/Anaconda/Thesis/tiles/test/sets/y_test_vector_tile_dict_size_3748_04-05-2020-1146.pickle", 'rb') as handle:
           test_vector_tile_dict = pickle.load(handle)


X_test =  np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/" + "right_X_test_800_04-18-2020-1133.npy" )
y_test =  np.load("c:/Users/Szedlák Barnabás/Anaconda/Thesis/model/vectors/" + "right_y_test_800_04-18-2020-1133.npy" )
with open("c:/Users/Szedlák Barnabás/Anaconda/Thesis/tiles/test/sets/right_y_test_vector_tile_dict_size_800_04-18-2020-1133.pickle", 'rb') as handle:
           test_vector_tile_dict = pickle.load(handle)


#------------------------------------------------------------------------------
# 3. STEP
# Building the cnn model


####################################################### IMPORTANT SETTINGS!!!!!
learning_rate = 0.002
batch_size = 100
epoch_no = 10
####################################################### IMPORTANT SETTINGS!!!!!

optimizer = Adam(lr=learning_rate)

model = Unet(backbone_name='efficientnetb7', 
             classes = len(mask_label_dict.keys()), 
             activation = 'softmax', 
             encoder_weights = 'imagenet', 
             encoder_freeze = True)    # freezing weights as pre-trained weights are used

model.compile(optimizer, 
              loss = JaccardLoss(per_image = False), 
              metrics = ['categorical_accuracy', IOUScore(per_image = False, threshold = 0.5)])

# creating generators for the image augmentation
train_generator = backroom.UnetSequence(X_train, y_train_multidim, batch_size, augmentations = backroom.train_augmentation) 
test_generator = backroom.UnetSequence(X_test, y_test_multidim, batch_size, augmentations = None)


start_time = time.time() # measuring modelling time

# basic .fit method
model.fit(X_train, y_train_multidim, epochs = 2, batch_size = batch_size, validation_data = (X_test, y_test_multidim)) 

set_trainable(model, recompile = False) # Set all layers of model trainable, so that encode_freeze is lifted. Recompile = True does not work with Tensorflow 2.0
model.compile(optimizer, 
              loss = JaccardLoss(per_image = False), 
              metrics = ['categorical_accuracy', IOUScore(per_image = False, threshold = 0.5)])

# fit_generator method for image augmentation
model.fit_generator(train_generator, 
                    validation_data=test_generator, 
                    steps_per_epoch=len(X_train) // batch_size, 
                    validation_steps=len(X_test) // batch_size, 
                    epochs=epoch_no, 
                    callbacks=backroom.callbacks)

elapsed_time = time.time()-start_time # measuring modelling time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) # beautifying time format




#------------------------------------------------------------------------------
# 4. STEP
# Prediction 

model.load_weights(SSD_path + "model/weights/fin_M1_unet_best.h5")
y_pred = model.predict(X_test)


# croping image vector extensions
y_pred = y_pred[:,16:80,16:80,:]
y_test = y_test[:,16:80,16:80]  # original y_test is stored above


# dimension reduction (uses 54x64 vector size)
y_pred_twodim = backroom.dimension_reduction(y_pred, mask_label_dict)




#------------------------------------------------------------------------------
# 5. STEP
# Validation 


results = backroom.get_metrics(y_test, y_pred_twodim)

print(results['weighted_score'],
      results['unweighted_score'],
      results['label_2'],
      results['label_3'],
      results['label_4'],
      results['label_5'],
      results['label_7'],
      results['label_8'],
      results['label_9'],
      results['label_10'],
      results['label_11'],
      results['label_17'],
      results['label_18'])


#------------------------------------------------------------------------------
# 6. STEP
# Documentation 


# creating RGB vectors
y_pred_RGB = backroom.RGB_coloriser(y_pred_twodim, mask_label_dict)  
y_test_RGB = backroom.RGB_coloriser(y_test, mask_label_dict)  


# clearing existing prediction tiles from folder
backroom.clear_folder(SSD_path + "model/test_tiles/pred/*")
backroom.clear_folder(SSD_path + "model/test_tiles/mask/*")


# saving tile files
backroom.vector_to_tif(y_pred_twodim, test_vector_tile_dict, 'prediction', RGB=False)
backroom.vector_to_tif(y_pred_RGB, test_vector_tile_dict, 'prediction', RGB=True)
backroom.vector_to_tif(y_test_RGB, test_vector_tile_dict, 'test_mask', RGB=True)


# creating mosaics for prediction
backroom.mosaic('test','pred', RGB = True) 
#backroom.mosaic('test','pred', RGB = False)
backroom.mosaic('test','mask', RGB = True) 
#backroom.mosaic('test','mask', RGB = False)
backroom.mosaic('test','raster', RGB = False) 


# train vectors processing (optional)
y_train_RGB = backroom.RGB_coloriser(y_train, mask_label_dict)  
backroom.vector_to_tif(y_train_RGB, train_vector_tile_dict, 'train_mask', RGB=True)
backroom.mosaic('train','raster', RGB = False) 
backroom.mosaic('train','mask', RGB = True) 


