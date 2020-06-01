# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:13:05 2020

@author: Szedlák Barnabás
"""

from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio import windows
from itertools import product
import pickle
from collections import defaultdict


#------------------------------------------------------------------------------
# 1. STEP
# Defining function to create tiles


# updated fuction from the sample DHI project
def get_tiles(datasource, width=64, height=64, fold=1, extension=16):
    """Function to create the product matrix for the tiles from the big picture.
    Width & Height corresponds to the tile dimensions, whereas fold refers to the overlap of the tiles.
    1 = no overlap, 2 = overlapping by a half tile, and so on."""
    ncols, nrows = datasource.meta['width'], datasource.meta['height']
    offsets = product(range(0, ncols, width//fold), range(0, nrows, height//fold))  # control the level of overlap
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
   
    # defining window and transfor functions (original and extended)
    for col_off, row_off in offsets:
        # creating the original tile
        window_original = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform_original = windows.transform(window_original, datasource.transform)        
        
        # creating the extended boundary tile
        col_off_extended = col_off - extension
        row_off_extended = row_off - extension
        width_extended = width + extension*2
        height_extended = height + extension*2

        # windowing with the extended tile
        window_extended = windows.Window(col_off = col_off_extended, 
                                         row_off = row_off_extended, 
                                         width = width_extended, 
                                         height = height_extended).intersection(big_window)

        transform_extended = windows.transform(window_extended, datasource.transform)
                
        yield window_original, transform_original, window_extended, transform_extended
        


#------------------------------------------------------------------------------
# 2. STEP
# Initialising the parameters and listing all the image files

datadir = "train_test_split/"
SSD_path = "c:/Users/Szedlák Barnabás/Anaconda/Thesis/"

raster_file_list = ["left_input_train",
                    "midnorth_input_train",
                    "midcity_input_train",
                    "midsouth_input_train",
                    "right_input_train",
                    "island_input_train",
                    "left_input_test",
                    "midnorth_input_test",
                    "midcity_input_test",
                    "midsouth_input_test",
                    "right_input_test",
                    "island_input_test"]
mask_file_list = ["left_mask_train",
                  "midnorth_mask_train",
                  "midcity_mask_train",
                  "midsouth_mask_train",
                  "right_mask_train",
                  "island_mask_train", 
                  "left_mask_test",
                  "midnorth_mask_test",
                  "midcity_mask_test",
                  "midsouth_mask_test",
                  "right_mask_test",
                  "island_mask_test"]



gen_label_dist_dict = {} # dictionary to map the label disctribution for the whole (general) dataset
gen_label_dist_dict = defaultdict(lambda: 0, gen_label_dist_dict) # in defaultdict all values are set to be 0 unless stated otherwise
temp_dict = {} # temporary dictioary to store local label distribution dictionaries
total = [] # counting total number of pixels, in a form of a list to have image-wise overview

tile_width = 64
tile_height = 64
fold = 1
extension = 16 # per side as Unet imput must be divisible by 32!!!



#------------------------------------------------------------------------------
# 3. STEP
# Creating the tiles

# indetions are required and important to keep both raster and mask files open while reading the small windows with the get_tiles() details

for i in range(0,12):

    folder = raster_file_list[i].split("_")[2] # deciding whetehr file is train or test
   
    # checking to make sure right train-test raster-mask files are paired
    assert (raster_file_list[i][0] == mask_file_list[i][0]) and (raster_file_list[i][2] == mask_file_list[i][2]),"Raster-Mask Train-Test pairing failed"
       
    raster_file = datadir + raster_file_list[i] + ".tif"
    mask_file = datadir + mask_file_list[i]  + ".tif"
    
    print("Raster file to process:\t" + raster_file)
    print("Mask file to process:\t" + mask_file)
    
    # reading the raster file
    with rasterio.open(raster_file) as raster:
        raster_meta = raster.meta.copy()
        raster_nodata = raster_meta['nodata']
        raster_height = raster_meta['height']
        raster_width = raster_meta['width']    
        
        # acquiring the maximum number of tiles possible, which is only needed for printing out the process status.
        # map_matrix is a matrix to store where each tile is situated in the grid of tiles
        if fold > 1:
            label_max_count = ceil(raster_height/tile_height) * ceil(raster_width/tile_width) + ceil(raster_height/tile_height) * ceil(raster_width/tile_width) * (1/fold)
            map_matrix_y = ceil(raster_height/tile_height) + ceil(raster_height/tile_height)*(1/fold)
            map_matrix_x = ceil(raster_width/tile_width) + ceil(raster_width/tile_width)*(1/fold)
    
        else:
            label_max_count = ceil(raster_height/tile_height)*ceil(raster_width/tile_width)
            map_matrix_y = ceil(raster_height/tile_height)
            map_matrix_x = ceil(raster_width/tile_width)
        
        map_matrix = np.empty(label_max_count, dtype = np.int)
        map_matrix.fill(-1)
        
        # reading the mask file
        with rasterio.open(mask_file) as mask:
            mask_meta = mask.meta.copy()
            mask_nodata = mask_meta['nodata']
         
            # once both raster and mask files are loaded, we can create the tiles
            tile_counter = 0 # counting all the tiles possible
            file_counter = 0 # counting all the tiles saved after the condition checks
            loc_label_dist_dict = {} # dictionary to map the label disctribution for the given (local) image file
            loc_label_dist_dict = defaultdict(lambda: 0, loc_label_dist_dict) # all values are set to be 0 unless stated otherwise
            
            # instead of storing actual image data, these dicionaries store there meta information
            tile_dictionary = {}
            tile_extended_dictionary = {}
           
            # calling the get_tiles() function
            for window_original, transform_original, window_extended, transform_extended in get_tiles(raster, tile_width, tile_height, fold, extension): 
                """since raster and mask files are identical in dimensions, 
                get_tiles() is enought to be called for the raster file,
                and to use the returned small winows for raster and mask files"""
                raster_data = raster.read(window = window_extended)
                mask_data = mask.read(window = window_extended)

                # if full sized small window (and full size extended windom) could have been created (thus smal window is not at the edge of the input image)
                if window_original.width ==  tile_width \
                and window_original.height == tile_height \
                and window_extended.width == tile_width + extension*2 \
                and window_extended.height == tile_width + extension*2:
                    
                    """ FILTERING INVALID IMAGES MUST HAPPEN HERE!!!"""
                    
                    if 1 in mask_data: # mask data has NO DATA (== 1)
                        pass
                    
                    elif len(raster_data[np.where(raster_data[:,:,:] == 0)]) / np.ma.size(raster_data) > 0.1 : # if [0,0,0] values are more than 10% of a tile
                        pass
                    
                    else: # wiriting both raster and mask tiles
                        """
                        This code saves the tiles individually as .tif file. This is time and space consuming.
                        
                        # updating meta data for raster and mask files for original tile, not in use currenty
                        raster_meta['transform'] = transform_original
                        raster_meta['width'] = window_original.width
                        raster_meta['height'] = window_original.height
                        mask_meta['transform'] = transform_original
                        mask_meta['width'] = window_original.width
                        mask_meta['height'] = window_original.height
                
                        with rasterio.open(SSD_path + "tiles/" + folder + "/raster/" + raster_file_list[i].split("_")[0] + "_" + raster_file_list[i].split("_")[1] + "/" + \
                            str(file_counter) + "_" + raster_file_list[i] + "_tile.tif", 'w', **raster_meta) as out_raster:
                            out_raster.write(raster_data)
                        with rasterio.open(SSD_path + "tiles/" + folder + "/mask/" + mask_file_list[i].split("_")[0] + "_" + mask_file_list[i].split("_")[1] +  "/" + \
                            str(file_counter) + "_" + mask_file_list[i] + "_tile.tif", 'w', **mask_meta) as out_mask:
                            out_mask.write(mask_data)
                        """
                        
                        # instead, let's store only meta information of the possible tiles
                        map_matrix[tile_counter] = file_counter
                        tile_dictionary[file_counter] = [window_original, transform_original] # dictionary for original tile counts and window/transform values
                        tile_extended_dictionary[file_counter] = [window_extended, transform_extended] # dictionary for extended tile counts and winfow/transform values
                        file_counter += 1
                        
                        # this section was added to calculate the label distribution for the vector set
                        for label in np.unique(mask_data[0,:,:]): # acquiring unque labels in the tile
                            gen_label_dist_dict[label] += len(np.where(mask_data[0,:,:] == label)[0]) # counting the coordinates of akin pixels for all images
                            loc_label_dist_dict[label] += len(np.where(mask_data[0,:,:] == label)[0]) # counting the coordinates of akin pixels for current image
                    
                    
                tile_counter += 1
                
                # 'in process' print out
                if tile_counter % 1000 == 0:
                    print("%s of %s tiles are processed" % (tile_counter, label_max_count))
                    print("%s files are created" % file_counter)
                            
    # reshaping map_matrix to match exactly the image file dimensions  
    map_matrix = map_matrix.reshape((map_matrix_x, map_matrix_y))
    map_matrix = map_matrix.swapaxes(0,1)
    
    
    # 'final' print out
    print("--------------------------------------------------------------")
    print("%s,\t%s" %(raster_file_list[i], mask_file_list[i]))
    print("%s of %s tiles are processed" % (tile_counter, label_max_count))
    print("%s files are created" %file_counter)
    print("Mapping matrix shape: " + str(map_matrix.shape))
    
    # saving map_matrix of the tiles
    np.savetxt(SSD_path + "tiles/" + folder + "/" + raster_file_list[i].split("_")[0] + "_tile_mapping_" + folder + ".csv", map_matrix, fmt = '%i', delimiter = ',')
    print("Mapping matrix is saved: " + raster_file_list[i].split("_")[0] + "_tile_mapping.csv")
   
    # saving the tile_dictioanries    
    with open(SSD_path + "tiles/" + folder + "/" + raster_file_list[i].split("_")[0] + "_tile_dictionary_" + folder + ".pickle", 'wb') as handle_original:
        pickle.dump(tile_dictionary, handle_original, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tile dictionary is saved: " + folder + "_" + raster_file_list[i].split("_")[0] + "_tile_dictionary.pickle")
    
    # saving the tile_extended_dictioanries 
    with open(SSD_path + "tiles/" + folder + "/" + raster_file_list[i].split("_")[0] + "_tile_extended_dictionary_" + folder + ".pickle", 'wb') as handle_extended:
        pickle.dump(tile_extended_dictionary, handle_extended, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tile extended dictionary is saved: " + folder + "_" + raster_file_list[i].split("_")[0] + "_tile_extended_dictionary.pickle")
    
    print() # empty line
    print() # empty line
    
    # counting the total number of pixels
    loc_total = np.prod([file_counter, tile_height + extension*2, tile_width + extension*2]).astype('int64')
    total.append(loc_total) # sill inside the for loop, sotring values for each input image    
    # storing the pixel distributions for the given mask image
    loc_label_dist_dict = {k: round(v / loc_total * 100,4) for k,v in loc_label_dist_dict.items()} # converting local pixel counts to % outside of the loop
    temp_dict[mask_file_list[i]] = loc_label_dist_dict
  


#------------------------------------------------------------------------------
# 4. STEP
# Creating the label distribution overviews
    
    
# calculating and saving the label percentages for the whole dataset (now, this is Big Data indeed)
gen_label_dist_dict = {k: round(v / sum(total) * 100,4) for k,v in gen_label_dist_dict.items()} # converting general pixel counts to % outside of the loop
with open(SSD_path + "tiles/general_label_distribution_dictionary.pickle", 'wb') as handle:
        pickle.dump(gen_label_dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("General Label disctribution is saved, as tiles/general_label_distribution_dictionary.pickle")


# creating general overview for each mask image label distribution with a dataframe
mask_label_dict = {2: 'Road(2)',
                   3: 'Building(3)',
                   4: 'Water(4)',
                   5: 'Rail(5)',
                   7: 'Paved(7)',
                   8: 'Rock(8)',
                   9: 'Gravel(9)',
                   10:'Forest(10)',
                   11:'Other vegetation(11)',
                   17:'Forest 17)',
                   18:'Other vegetation(18)'}

# creating summary distribution dataframe.
distribution_df = pd.DataFrame.from_dict(gen_label_dist_dict, orient ='index', columns = ['overall'])
for key in temp_dict.keys():
    distribution_df[key] = pd.DataFrame.from_dict(temp_dict[key], orient = 'index')

# creating summary distribution plot
figure, axes = plt.subplots(3, 4, sharex = True, sharey=True)
figure.suptitle('Individual label distributions\n' , fontweight='bold')
for index, column in enumerate(distribution_df.drop(columns = ['overall']).columns):
    col = index % 4
    row = index // 4
    ax = axes[row][col]
    distribution_df[['overall', column]].plot(ax = ax, kind = 'bar', title = column, legend = False)
    ax.tick_params(axis = 'x', labelrotation = 0)
    ax.grid(True, axis ='y')
    handles, _ = ax.get_legend_handles_labels()
figure.legend(handles, ['overall', 'mask file'] , loc='best')
figure.subplots_adjust(bottom=0.06)
figure.text(0.08, 0.5, 'presence %', va='center', rotation='vertical')
figure.text(0.5, 0.01, 'labels')
print("Individual disctributions plot is created, save if needed.")

# saving the general lable distributions as a csv, too
distribution_df['labels'] = pd.DataFrame.from_dict(mask_label_dict, orient = 'index')
distribution_df.set_index('labels', inplace = True)
distribution_df.to_csv(SSD_path + "tiles/individual_label_distributions.csv")
print("Individual disctributions .csv is saved to tiles/individual_label_disctributions.csv")



#------------------------------------------------------------------------------
# 5. STEP
# Testing
# Checking if tiler work properly (portion)
                
                
with rasterio.open("tiles/train/8064left_raster_train_tile_.tif") as test:
    test_meta = test.meta.copy()
    test_nodata = test_meta['nodata']
    test_data = test.read()

with rasterio.open("tiles/train/3640left_raster_train_tile_.tif") as test2:
    test2_meta = test2.meta.copy()
    test2_nodata = test2_meta['nodata']
    test2_data = test2.read()    
    
    
if (test_data.mean() == 0): # condition to check
    print("ok")
else:
    print("not ok")

