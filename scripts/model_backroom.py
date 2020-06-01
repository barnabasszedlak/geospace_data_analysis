# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:23:59 2020

@author: Szedlák Barnabás
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import rasterio
import pickle
import rasterio.merge
import glob
import os

from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard # EarlyStopping,

from albumentations import (Compose, HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate, RandomBrightnessContrast)  
                            #RandomCrop, RandomGamma, IAAEmboss, Blur, OneOf, ElasticTransform, GridDistortion, OpticalDistortion, ToFloat

from sklearn.metrics import jaccard_score



###################
# Functions
        

# function to measure the Jaccard index - depricated!!!
def iou_metric(y_test, y_pred):
    """ Function copiend from "Tutorial.ipynb" as it is, to measure 
    the Intersection-Over-Union (Jaccard Index) for the test and prediction vectors."""
    
    temp1 = np.histogram2d(y_test.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))

    intersection = temp1[0]

    area_true = np.histogram(y_test,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    union = union[1:,1:]
    union[union == 0] = 1e-9
    
    iou = (intersection / union)[0,0]
        
    return iou




def distribution_inspector(vector_label_dist_dict, mask_label_dict, vector_tile_dict):
    """
    Fuction to draw the label distribution for the WHOLE dataset and the generated vector set, in order to compare them.
    
    Parameters
    ----------
    vector_label_dist_dict : dictionary
        Contains the % values for each label in the vector set.
    mask_label_dict: dictionary for mask labels and their names.

    Returns
    -------
    Ax figure is plotted and summary table is displaced in the console.

    """
    
    print("distribution_inspector is commencing...")
    
    with open("c:/Users/Szedlák Barnabás/Anaconda/Thesis/tiles/general_label_distribution_dictionary.pickle", 'rb') as handle:
        rawimage_label_dist_dict = pickle.load(handle)
    
    chart_df = pd.DataFrame.from_dict(mask_label_dict, orient ='index', columns = ['RGB','Name'])
    chart_df['raw_data'] = pd.DataFrame.from_dict(rawimage_label_dist_dict, orient ='index')
    chart_df['vector_set'] = pd.DataFrame.from_dict(vector_label_dist_dict, orient = 'index')
    chart_df.drop(columns = ['RGB'], inplace = True)
    chart_df.set_index('Name', inplace = True)
    
    
    ax = chart_df.plot(kind='bar',figsize=(8, 6))
    plt.xticks(rotation=30)
    plt.grid(axis = 'y')
    plt.xlabel('mask labels', fontweight='bold')
    plt.ylabel('presence [%]', fontweight='bold')
    plt.title('Label distributions\n\n' , fontweight='bold')
    
    plt.tight_layout(pad=0.5)
    
    for i in ax.patches:
        if i.get_height() != 0:
            ax.text(i.get_x()+0.01, i.get_height()+0.5,horizontalalignment='left', s = round(i.get_height(),2), fontsize=12, color='black')
        else:
            ax.text(i.get_x()+0.01, i.get_height()+0.5,horizontalalignment='left', s = round(i.get_height(),2), fontweight='bold', fontsize=12, color='red')
    
    support_text1 = "Vector size: "  + str(sum([len(v) for k, v in vector_tile_dict.items()]))
    support_text2 = "Vector is built from: " + str({k: len(v) for k, v in vector_tile_dict.items()})
    plt.text(0.07, 0.05, support_text1, fontsize=10, fontweight='bold', transform=plt.gcf().transFigure)
    plt.text(0.07, 0.025, support_text2, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.865, 0.025, datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"),fontsize=10, transform=plt.gcf().transFigure)
    
    print("distribution_inspector >>> figure created")
    return chart_df




def get_metrics(y_true, y_pred, mask_labels = [2,3,4,5,7,8,9,10,11,17,18]):
    """ Function to measure the performance of the model on the test set, using the Jaccard score.

    Parameters
    ----------
    y_true : ground truth vector
        DESCRIPTION.
    y_pred : prediction vector. Dimensions must match that of ground truth vector.
        DESCRIPTION.
    mask_labels : mask labels used in the prefiction., optional
        The default is [2,3,4,5,7,8,9,10,11,17,18].

    Returns
    -------
    result_dict : dictionary
        Jaccard score for individual labels, wighted average and simple average.
    """
    
    
    print("get_metrics is commencing...")
    # IOU measure for each label:
    result_dict = {}
    for index, key in enumerate(mask_labels):
        result_dict['label_' + str(key)] = round(jaccard_score(y_true.flatten(), y_pred.flatten(), labels = mask_labels, average = None)[index],4)

    # IOU measure for weighted label average:
    result_dict['weighted_score'] = round(jaccard_score(y_true.flatten(), y_pred.flatten(), labels = mask_labels, average = 'weighted'),4)
    
    # IOU measure for each label average:
    result_dict['unweighted_score'] = round(jaccard_score(y_true.flatten(), y_pred.flatten(), labels = mask_labels, average = 'macro'),4)
    
    print("get_metrics >>> metrics are returned")
    return result_dict




# function to generate tif file
def vector_to_tif(vector, vector_tile_dict, vector_type, RGB = False):
    """Function to generate tif files from vectors Currently only works with test and prediction vectors.
    
     - vector = the input vector that maches the tile dictionary (y_pred or y_test),
     - tile_map_dict = the dictionary containing the mask file name and tile numbers,
     - vector_type = mask or prediction,
     - RGB = True, if input vector is RGB vector.
   """
    
    print("vector_to_tif is commencing...")
    SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/"
    datadir = "train_test_split/" 
    
    
    if RGB:
        rgb_suffix = "_RGB"
    else:
        rgb_suffix = ""
        
    # selecting the matching tile_dictionary
    if vector.shape[1] == 64:
        tile_dict_name = "_tile_dictionary_"
    elif vector.shape[1] == 96:
        tile_dict_name = "_tile_extended_dictionary_"
    else:
        raise ValueError('The vector dimensions do not match tile_dictionary dimensions')
        
    print("Vector shape: " + str(vector.shape) + " therefore: " + tile_dict_name)
    
    # dictionary to map vector_type settings. Dictionary is used for further possible improvements
    mapping_dict = {'prediction': ['test','mask','pred'], # input image folder / type of file / result folder type
                    'test_mask': ['test','mask','mask'],
                    'train_pred':['train','mask','pred'],
                    'train_mask':['train','mask','mask']} # this last one must be deleted, used only for verification
    
    index_counter = 0 # main count for the vector index
    for key in vector_tile_dict.keys():   
        
        with open(SSD_path + "tiles/" + mapping_dict[vector_type][0] + "/" + key + tile_dict_name + mapping_dict[vector_type][0] + ".pickle", 'rb') as handle:
            tile_dictionary = pickle.load(handle)
        
        tile_file_list = vector_tile_dict[key] # acquiring the tile numbers for the given key in the loop
        image_file = datadir + key + "_" + mapping_dict[vector_type][1] + "_" + mapping_dict[vector_type][0] + ".tif" # acquiring the matching image file
        
        print("Tile_dict: \t" + SSD_path + "tiles/" + mapping_dict[vector_type][0] + "/" + key + tile_dict_name + mapping_dict[vector_type][0] + ".pickle")
        print("Image_file: " + datadir + key + "_" + mapping_dict[vector_type][1] + "_" + mapping_dict[vector_type][0] + ".tif")
         
        with rasterio.open(image_file) as image:
            image_meta = image.meta.copy()
            
            for index, file_count in enumerate(tile_file_list):
                print(str(index_counter + 1) + ": " + str(index + 1) + " / " + str(len(tile_file_list)) + " (" + key+ ") - " + "file_count: " + str(file_count))
                
                window, transform = tile_dictionary[file_count]
                # updating meta data for raster and mask files
                image_meta['transform'] = transform
                image_meta['width'] = window.width
                image_meta['height'] = window.height
                
                if RGB: # there is a color dimension
                    image_data= vector[index_counter, :, :, :].astype('uint8')
                    image_data = np.swapaxes(image_data, 1, 2)
                    image_data = np.swapaxes(image_data, 0, 1)
                    image_meta['count'] = 3 # updating band number

                else: # mask and prediction non RGB vectors have 3 dimensions only
                    image_data = vector[index_counter,:,:].astype('uint8')
                    image_data = image_data.reshape(1,vector.shape[1], vector.shape[2])
                    
                    
                # saving the new predicted tiles to the model/tiles folder as "predicted"
                with rasterio.open(SSD_path + "model/" + mapping_dict[vector_type][0] + "_tiles/" + mapping_dict[vector_type][2] + "/" + str(file_count) + "_" + \
                                   key + "_" + mapping_dict[vector_type][2] + rgb_suffix + "_tile.tif", 'w', **image_meta) as out_image:
                    out_image.write(image_data) 
                index_counter += 1 # increasing the index counter as a vector is saved out
    
    print("pred_to_fif >>> predicted %s tif files are succesfully created from vector" % rgb_suffix)




# function to create a big mosaic fron the test tiles
def mosaic(tile_type, tiles_folder, RGB = False):
    """Function to build mosaic from tile files. 
    
     - tile_type = train or test
     - test_tiles_folder = the folder in which the test tiles are present, which are created with the X_test, y_test vectors for the model. 
     As for now: raster / mask / test
     - RGB = RGB or non RGB files to be considered
    """
    
    print("mosaic is commencing...")
    SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/"
    resultdir = SSD_path + "model/" + tile_type + "_tiles/" + tiles_folder + "/*.tif"
    src_files_to_mosaic = []
    
    try:
        if RGB: # creating mosaic for RGB tiles
            rgb_suffix = "_RGB"
            for pred_tile in glob.glob(resultdir):
                if "RGB" in pred_tile:
                    src = rasterio.open(pred_tile)
                    src_files_to_mosaic.append(src)
            else:
                pass
            
        else: #creating mosaic for non_RGB tiles
            rgb_suffix = ""
            for pred_tile in glob.glob(resultdir):
                if "RGB" in pred_tile:
                    pass
                else:
                    src = rasterio.open(pred_tile)
                    src_files_to_mosaic.append(src)
    
        mosaic, out_trans = rasterio.merge.merge(src_files_to_mosaic)
    
    
    except IndexError:
        print("No files found")
        return
    
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,})
   
    with rasterio.open(SSD_path + "model/" + tile_type + "_tiles/mosaic/" + tiles_folder + rgb_suffix +"_mosaic.tif", "w", **out_meta) as dest:
        dest.write(mosaic)
    
    print("mosaic >>> Mosaic for %s folder %s is done" % (tiles_folder, rgb_suffix))




# function to create 4th dimesions for distinct mask labels
def mask_to_multidim(y_vector, mask_labels = [2,3,4,5,7,8,9,10,11,17,18]):
    """Function to map each mark label into different dimensions in the y vector sets (y_train, y_test)
    
    - y_vector = y_train / y_test vectors
    - mask_labels = list of mask lable numbers to consider. By default it equals to [2,3,4,5,7,8,9,10,11,17,18]"""
    
    print("mask_to_multidim is commencing...")
    # initialising the multidimensional vector
    multidim_vector = np.empty((y_vector.shape[0], y_vector.shape[1], y_vector.shape[2], len(mask_labels)), dtype=np.int8)
    
    # looping through the input vector using the mask_labes
    for index, label in enumerate(mask_labels):
        temp = y_vector.copy()
        temp[np.where(temp != label)] = 0
        temp[np.where(temp == label)] = 1
        multidim_vector[:,:,:,index] = temp
        
    print("mask_to_multidim >>> multidimensional vector is succesfully created for %s label dimensions" %len(mask_labels))
    
    return multidim_vector




# function to map the mask labels dimensions back to pixel values
def dimension_reduction(y_vector, mask_label_dict):
    """Function to loop over the dimensions of the prediction vector and map the mask labels back.
    The logic selects the maximum probability dimensions, thus mask label, for each pixel.
    
    - y_vector = y_pred,
    - mask_labels_dict = dictionary with the mask labes used in the prediction."""
    
    print("dimension_reduction is commencing...")
    same_prob_error = 0 # keeping track of the same probablility errors. Currenlty not in use but left here for later purposes
    y_vector_twodim = np.empty((y_vector.shape[0], y_vector.shape[1], y_vector.shape[2]), dtype=np.uint8)
    for index in range(y_vector.shape[0]):
        for i in range(y_vector.shape[1]):
            for j in range(y_vector.shape[2]):
                try:
                    y_vector_twodim[index,i,j] = list(mask_label_dict.keys())[int(np.where(y_vector[index,i,j,:] == max(y_vector[index,i,j,:]))[0])]
                except TypeError:
                    #print("original: ", y_vector[index,i,j,:])
                    #print("max: ", max(y_vector[index,i,j,:]))
                    #print("Issue :", np.where(y_vector[index,i,j,:] == max(y_vector[index,i,j,:])))
                    y_vector_twodim[index,i,j] = list(mask_label_dict.keys())[int(np.where(y_vector[index,i,j,:] == max(y_vector[index,i,j,:]))[0][0])]
                    same_prob_error += 1
                #print("selected: ", str(list(mask_label_dict.keys())[int(np.where(y_vector[index,i,j,:] == max(y_vector[index,i,j,:]))[0])]))
                
    print("dimesnion_reduction >>> prediction vector is sucessfully mapped back")
    
    return y_vector_twodim




# function to create RGB vectors
def RGB_coloriser(y_vector, mask_label_dict):
    """Function to create RGB vectors based on the massk label values and required colors 
    specified in the mask_label_dict.
    
    - y_vector = vector to be colorised. It must be 3 dimensional where first dimension is the stack (y_test or y_pred)
    - mask_label_dict = dictionary with the mask labes (and RGB colors) used in the prediction."""
    
    print("RGB_coloriser is commencing...")
    y_vector_RGB = np.empty((y_vector.shape[0], y_vector.shape[1], y_vector.shape[2], 3), dtype=np.uint8) # datatype is uint8 as so 0-255 for RGB
    for index in range(y_vector.shape[0]):
        for i in range(y_vector.shape[1]):
            for j in range(y_vector.shape[2]):
                y_vector_RGB[index,i,j,:] = np.array(mask_label_dict[y_vector[index,i,j]][0]) # extra index is needed as now not only RGB values but label names are stored
    
    print("RGB_coloriser >>> vector is sucessfully colorised to RGB values")
    
    return y_vector_RGB
       


# function to delete files from a folder
def clear_folder(path):
    """Function to clear the ALL content from a folder specified by the path.
    
    - path = relative path""" 
     
    print("clear_folder is commencing...")     
    files = glob.glob(path)
    for f in files:
        try:
            os.remove(f)
        except:
            pass
    
    print("clear_folder >>> Files are deleted from %s" %path)
    
    
    

# sequence to load data and apply augmentation
class UnetSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        data_index_min = int(idx*self.batch_size)
        data_index_max = int(min((idx+1)*self.batch_size, len(self.x)))  # maybe the data is smaller that a batch
        
        indexes = self.x[data_index_min:data_index_max] # these are not only "indeces", but the data itself! Really weird way to calculate length...
        this_batch_size = len(indexes) # just to be sure about the actual size of this batch
        
        X = np.empty((this_batch_size, y_size, x_size, 3), dtype=np.float32) # last dimension is changed from 6
        y = np.empty((this_batch_size, y_size, x_size, 11), dtype=np.uint8) # y last dimension is changed to 11 <<< change if less mask labels are to be used
        
        for i, sample_index in enumerate(indexes):
            x_sample = self.x[idx * self.batch_size + i, :, :, :] # code modified, so that each iteration uses a new original image vector
            y_sample = self.y[idx * self.batch_size + i, :, :, :] # extra dimension is added as labes are multidimensioned.
            if self.augment is not None:
                augmented = self.augment(image=x_sample, mask=y_sample)
                
                # new lines from the documentation
                #data = {"image": x_sample, "mask": y_sample}
                #augmented = self.augment(**data)
            
                image_augm = augmented['image']
                mask_augm = augmented['mask']
                X[i, :, :, :] = image_augm
                y[i, :, :, :] = mask_augm
            else:
                X[i, :, :, :] = x_sample
                y[i, :, :, :] = y_sample
        return X, y
    



    
###################
# Model parameters
        
    
SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/"
x_size = 96
y_size = 96


# Augmentation settings
train_augmentation = Compose([HorizontalFlip(p=0.5), # p = probalility of given augmentation feature happening
                              Transpose(p=0.5),
                              ShiftScaleRotate(p=0.25, rotate_limit=0),
                              RandomBrightnessContrast(p=0.5)], p = 1) # final p is the probabbilit of the whole
                              #VerticalFlip(p=0.5),
                              #RandomRotate90(p=0.5),

# reduces learning rate on slowing on plateau callback
learning_rate_reducer = ReduceLROnPlateau(factor=0.1,
                                          cooldown= 2,
                                          patience=5,
                                          verbose =1,
                                          min_lr=0.1e-5)
                                        # monitor = 
# model autosave callbacks
model_save = ModelCheckpoint(SSD_path + "model/weights/unet.h5", 
                             monitor='val_iou_score', 
                             mode='max', 
                             save_best_only=True, 
                             verbose=1,
                             period='epoch')


# tensorboard logging metric to display
logdir = "logs/fit/"
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# callback stacking
callbacks = [learning_rate_reducer, tensorboard_callback, model_save]




print("model_backroom.py script is imported")