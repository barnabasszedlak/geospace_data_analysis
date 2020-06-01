# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:15:34 2020

@author: Szedlák Barnabás
"""

import numpy as np
import pandas as pd
import rasterio
import random
import pickle
from model_backroom import clear_folder, mosaic
from datetime import datetime
from collections import defaultdict


#------------------------------------------------------------------------------
# 2. STEP
# Main function to iterate over the input images in order radnomly generate vector sets


def random_vectorizer(image_dict, size, neighbor_no, folder):  
    """Function to generate the X and y vectors for the training and the test sets. 
    X vectors are built by randomly picking a tile in the tile grid and mapping its neighbors to the extend specified as an input parameter (neighbor_no),
    for each and every input raster file specified in the image_dict input argument. y vectors are built by mapping the matching mask tiles."""
    
    SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/"
    datadir = "train_test_split/" 
    
    index_offset = 0 # used to build tiles from different file into a sinlge vectorset
  
    # initializing return objects
    X_vector = np.empty((size, 96, 96, 3), dtype=np.float32)
    y_vector = np.empty((size, 96, 96), dtype=np.int8)
    vector_tile_file_dict = {} # dictionary to map tile file count for the vectors created
    label_dist_dict = {} # dictionary to map the label disctribution in the generated sets
    label_dist_dict = defaultdict(lambda: 0, label_dist_dict) # in defaultdict all values are set to be 0 unless stated otherwise
    
    # looping over the input images
    for key, value in image_dict.items(): 
        print() # empty line
        print("-------------------------------------------------------------------")
        print("key: %s |\tproportion: %s" %(key, value))
        
        # reading the tile_mapping csv file, important not to parse the first line as header!
        matrix_df = pd.read_csv(SSD_path + "tiles/"+ folder + "/" + key + "_tile_mapping_" + folder + ".csv", header = None, engine = 'python') 
        # reading the tile_dictioanry pickle file
        with open(SSD_path + "tiles/" + folder + "/" + key + "_tile_extended_dictionary_" + folder + ".pickle", 'rb') as handle:
            tile_dictionary = pickle.load(handle)
    
        print("Matrix_df: \t" + SSD_path + "tiles/" + folder + "/" + key + "_tile_mapping_" + folder + ".csv")
        print("Tile_dict: \t" + SSD_path + "tiles/" + folder + "/" + key + "_tile_extended_dictionary_" + folder + ".pickle")
    
        tile_file_list = []
        counter = 0
        
        # selecting the tiles randomly and picking up their neighbors
        while counter < value:
            tile_file_count_random = int(random.uniform(0,matrix_df.values.max()+1)) # generating random tile_file counts using uniform distribution 
            x_cord = int(np.where(matrix_df == tile_file_count_random)[0]) # acquiring the x,y coordinates of the random file count in the mapping dataframe
            y_cord = int(np.where(matrix_df == tile_file_count_random)[1])
        
            for i in range(neighbor_no): # this loop also takes care for the first tile, which was found by the random value
                for j in range(neighbor_no):
                    try:
                        tile_file_count_neighbor = matrix_df.iloc[x_cord + i, y_cord + j]
                    
                        if counter >= value:
                            break
                        elif (tile_file_count_neighbor == -1):
                            pass
                        else:
                            if tile_file_count_neighbor not in tile_file_list:
                                tile_file_list.append(tile_file_count_neighbor)
                                counter += 1
                            else: 
                                pass  
                    except IndexError: # pointint over the matrix boudnaries
                        #print("Out of boundary occured") 
                        pass
        
        # identifying raster and mask files for the tiles
        raster_file = datadir + key + "_input_" + folder + ".tif"
        mask_file = datadir + key  + "_mask_" + folder + ".tif"
        
        print("Reading the following input image files:")
        print("\tRaster file: " + raster_file)         
        print("\tMask file: " + mask_file)
        
        # opening raster file
        with rasterio.open(raster_file) as raster:
            raster_meta = raster.meta.copy()
            #raster_nodata = raster_meta['nodata']
            #raster_height = raster_meta['height']
            #raster_width = raster_meta['width'] 
        
            # opening mask file
            with rasterio.open(mask_file) as mask:
                mask_meta = mask.meta.copy()
                #mask_nodata = mask_meta['nodata']
                    
                # looping over the tiles in the image file, using the meta information stored about them
                for index, file_count in enumerate(tile_file_list):
                    print(str(index_offset + 1) + ": " + str(index + 1) + " / " + str(len(tile_file_list)) + " (" + key + ") - " + "file_count: " + str(file_count))
                
                    window, transform = tile_dictionary[file_count]
                    
                    raster_data = raster.read(window=window)
                    raster_data = np.swapaxes(raster_data, 0, 1)
                    raster_data = np.swapaxes(raster_data, 1, 2)
                    mask_data = mask.read(window=window)
            
                    X_vector[index_offset, :, :, :] = raster_data
                    y_vector[index_offset, :, :] = mask_data[0, :, :]
                    
                    # this section was added to calculate the label distribution for the vector set
                    for label in np.unique(mask_data[0,:,:]): # acquiring unque labels in the tile
                        label_dist_dict[label] += len(np.where(mask_data[0,:,:] == label)[0]) # counting the coordinates of matching pixels
                    
                    index_offset += 1 # stepping indeyes in the X_vector and we populate it with the input image data
                    
                    
                    
                    # saving the test tiles for verification purposes (optional)
                    if folder == 'test':
                        
                        target_dest = {'train': 'train_tiles',
                                       'test': 'test_tiles'}
                        
                        # updating meta data for raster and mask files
                        raster_meta['transform'] = transform
                        raster_meta['width'] = window.width
                        raster_meta['height'] = window.height
                        mask_meta['transform'] = transform
                        mask_meta['width'] = window.width
                        mask_meta['height'] = window.height
                        
                        # Saving train raster tiles
                        raster_data = np.swapaxes(raster_data, 1, 2)
                        raster_data = np.swapaxes(raster_data, 0, 1)
                        
                        with rasterio.open(SSD_path + "model/" + target_dest[folder] + "/raster/" + str(file_count) + "_" + \
                            key + "_raster_tile.tif", 'w', **raster_meta) as out_raster:
                            out_raster.write(raster_data)
                        """
                        # Saving train mask tiles
                        with rasterio.open(SSD_path + "model/" + target_dest[folder] + "/mask/" + str(file_count) + "_" + \
                            key + "_mask_tile.tif", 'w', **mask_meta) as out_mask:
                            out_mask.write(mask_data)
                    """

        vector_tile_file_dict[key] = tile_file_list
        
    # calculating label distributions
    total = np.prod(y_vector.shape) # total number of pixes in the vector set
    label_dist_dict = {k: round(v / total * 100,4) for k,v in label_dist_dict.items()} # converting pixel counts to %

    return X_vector, y_vector, vector_tile_file_dict, label_dist_dict




#------------------------------------------------------------------------------
# 1. STEP
# Kernel function to call random_vectorizer(), provide the input values and take care of the output
    
    
def train_test_generator(train_size, train_neighbor_no, train_proportions_dict, test_size, test_neighbor_no, test_file_list, pre_norm = None):
    """Assembler function to call the "random-vectorizer" function and provide the required parameter.
    
    - train_size = the size of the training vector
    - train_neighbor_no = the number of neighboring tiles to be considered to the right and under the randonly selected one in the tile grid
    - train_proportion_dict = a dictionary with the input image names and their proportion in the training vector
    - test_size = the size of the test vector(s)
    - test_neighbor_no = the number of neighboring tiles to be considered to the right and under the randomly selected one in the tile grid 
    - test_dict = a dictionary with the input image names for wich test vector is to be created, individually"""

    SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/" # defined outside of the function too, as for now
    
    # clearing train and test files folders (remove previous tiles for better interpretarion)
    clear_folder(SSD_path + "model/test_tiles/raster/*")
    clear_folder(SSD_path + "model/test_tiles/mask/*")
    clear_folder(SSD_path + "model/train_tiles/raster/*")
    clear_folder(SSD_path + "model/train_tiles/mask/*")
    
    if pre_norm is None:
        # if this option is selected, composed train vector is generated. 
        # if not, previously generated train vector's norm vlaues need to be given to generate the test vectors
        
        # PART 1. - TRAIN VECTORS
        # -----------------------
        
        folder = "train"
        train_return_dict = {} # dictionary to store the train return vectors    
        # calculating input image train sizes with the proportions, and rounding up with the ceil function
        train_image_dict = {}
        for key, value in train_proportions_dict.items():
            if int(value * train_size / 100) > 0: # so that 0 proportioned input images are not even opened.
                train_image_dict[key] = int(value * train_size / 100)
            else:
                pass
        
        # adjusting train size based on the image proportions    
        true_train_size = sum(train_image_dict.values()) # in case of small values, due to the rouding original training size might reduced in reality
        X_train, y_train, train_vector_tile_dict, train_label_dist_dict = random_vectorizer(train_image_dict, true_train_size, train_neighbor_no, folder)
        
        # normalization with saving the training set minX and maxX into variables
        for band in range(X_train.shape[3]):
            minX = np.min(X_train[:,:,:,band])
            maxX =np.max(X_train[:,:,:,band])
            X_train[:,:,:,band] = (X_train[:,:,:,band] - minX) / (maxX - minX)
        
        # storing the train vectors
        train_return_dict['X_train'] = X_train
        train_return_dict['y_train'] = y_train
        train_return_dict['train_vector_tile_dict'] = train_vector_tile_dict
        train_return_dict['train_label_dist_dict'] = train_label_dist_dict
        train_return_dict['train_norm_minX_maxX'] = [minX, maxX]
        
        
        # saving train vectors
        np.save(SSD_path + "model/vectors/X_train_size_" + str(true_train_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), X_train)
        np.save(SSD_path + "model/vectors/y_train_size_" + str(true_train_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), y_train)
        # also saving the normalization values for future purposes
        np.save(SSD_path + "model/vectors/vector_norms/minXmaxX_size_" + str(true_train_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), [minX, maxX])
        # saving the tile dictionary with pickle
        with open(SSD_path + "tiles/" + folder + "/sets" + "/" + "y_train_vector_tile_dict_size" + "_" + str(true_train_size) + "_" + \
                  datetime.now().strftime("%m-%d-%Y-%H%M") + ".pickle", 'wb') as handle:
            pickle.dump(train_vector_tile_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)          
        # saving the tile label disctibution with pickle
        with open(SSD_path + "tiles/" + folder + "/sets" + "/" + "y_train_label_dist_dict_size" + "_" + str(true_train_size) + "_" + \
                  datetime.now().strftime("%m-%d-%Y-%H%M") + ".pickle", 'wb') as handle:
            pickle.dump(train_label_dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
    
    else:   
        # in this option, pretrained norms are provided by the function parameters. (they were calculated in previous runs, assuming there is no need for new training vector)
        minX = pre_norm[0]
        maxX = pre_norm[1]
        print('\n')
        print("WARNING: Previous normalization values are in use: " + str(minX), ", " + str(maxX))
        print('\n')
        print('WARNING: Composed train vector will not be created.')
        train_return_dict = {} # there must be someting to return in orer to fill the variable in the other script
        
    
    
    
    # PART 2. - COMPOSED TEST VECTORS
    # -------------------------------

    folder = "test" # updating folder, very important!!!
    test_return_dict = {} # dictionary to store  the test return values
    
    
    if pre_norm is None: # if this option is selected, composed test vector is generated. If not, only single image built test vectors are generated.
        # This part is used if test vectors are to be built together from the test files.
        test_image_dict = {}
        for key, value in train_proportions_dict.items():
            if int(value * test_size / 100) > 0: # so that 0 proportioned input images are not even opened.
                test_image_dict[key] = int(value * test_size / 100)
            else:
                pass
        # adjusting test size based on the image proportions
        true_test_size = sum(test_image_dict.values())
        X_test, y_test, test_vector_tile_dict, test_label_dist_dict = random_vectorizer(test_image_dict, true_test_size, test_neighbor_no, folder)
        for band in range(X_test.shape[3]):
            X_test[:,:,:,band] = (X_test[:,:,:,band] - minX) / (maxX - minX)
        
        test_return_dict["X_test"] = X_test
        test_return_dict["y_test"] = y_test
        test_return_dict["test_vector_tile_dict"] = test_vector_tile_dict
        test_return_dict["test_label_dist_dict"] = test_label_dist_dict
        
        # saving composed test vectors
        np.save(SSD_path + "model/vectors/X_test_size_" + str(true_test_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), X_test)
        np.save(SSD_path + "model/vectors/y_test_size_" + str(true_test_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), y_test)
        with open(SSD_path + "tiles/" + folder + "/sets" + "/" + "y_test_vector_tile_dict_size" + "_" + str(true_test_size) + "_" \
                  + datetime.now().strftime("%m-%d-%Y-%H%M") + ".pickle", 'wb') as handle:
            pickle.dump(test_vector_tile_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)          
        with open(SSD_path + "tiles/" + folder + "/sets" + "/" + "y_test_label_dist_dict_size" + "_" + str(true_test_size) + "_" \
                  + datetime.now().strftime("%m-%d-%Y-%H%M") + ".pickle", 'wb') as handle:
            pickle.dump(test_label_dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:    
        print('WARNING: Composed test vector will not be created.')
        print('\n')
        
        
    
    # PART 2.1 - SINGLE TEST VECTORS
    # ------------------------------
    
    for key in test_file_list: # should be chanfed to dictionary once more test vectors are to be merged
                
        temp_dict = {} # this hack is in place to have separate tet vectors for the test files. 
        temp_dict[key] = test_size

        vars()[key + "_X_test"], vars()[key + "_y_test"], vars()[key + "_test_vector_tile_dict"], vars()[key + "_test_label_dist_dict"]\
            = random_vectorizer(temp_dict, test_size, test_neighbor_no, folder)
        
        # normalization using the train vector min/max values (it only works if input images are from similar kind and time!!!)
        for band in range(vars()[key + "_X_test"].shape[3]):
            vars()[key + "_X_test"][:,:,:,band] = (vars()[key + "_X_test"][:,:,:,band] - minX) / (maxX - minX)
        
        # saving the test vectors as numpy vectors and as pickle dictionary
        np.save(SSD_path + "model/vectors/" + key + "_X_test" + "_" + str(test_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), vars()[key + "_X_test"])
        np.save(SSD_path + "model/vectors/" + key + "_y_test" + "_" + str(test_size) + "_" + datetime.now().strftime("%m-%d-%Y-%H%M"), vars()[key + "_y_test"])
        with open(SSD_path + "tiles/" + folder + "/sets" + "/" + key + "_y_test_vector_tile_dict_size" + "_" + str(test_size) + "_" \
                  + datetime.now().strftime("%m-%d-%Y-%H%M") + ".pickle", 'wb') as handle:
            pickle.dump(vars()[key + "_test_vector_tile_dict"], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(SSD_path + "tiles/" + folder + "/sets" + "/" + key + "_y_test_label_dist_dict_size" + "_" + str(test_size) + "_" \
                  + datetime.now().strftime("%m-%d-%Y-%H%M") + ".pickle", 'wb') as handle:
            pickle.dump(vars()[key + "_test_label_dist_dict"], handle, protocol=pickle.HIGHEST_PROTOCOL)
       
        # storing restuls in a dictionary that is returned by this fuction
        test_return_dict[key + "_X_test"] = vars()[key + "_X_test"]
        test_return_dict[key + "_y_test"] = vars()[key + "_y_test"]
        test_return_dict[key + "_test_vector_tile_dict"] = vars()[key + "_test_vector_tile_dict"]
        test_return_dict[key + "_test_label_dist_dict"] = vars()[key + "_test_label_dist_dict"]
        


    print("train_test_generator >>> Returned vectors are succesfully built.")
    return train_return_dict, test_return_dict




print("vectorizer.py script is imported")

