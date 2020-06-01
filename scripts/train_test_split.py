# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:17:55 2020

@author: Szedlák Barnabás
"""

import rasterio
from rasterio.windows import Window
from math import ceil



#------------------------------------------------------------------------------
# 1. STEP
# Individual modifications on the mid image as it is too big and most of it is forest


# mid image split is set to be 1/3
mid_split_size = 0.333

# reading the mid image RGB file to create windows
with rasterio.open('mid_RGB.tif') as mid_image:
    raster_metadata = mid_image.meta.copy()

# parameters
height = raster_metadata['height']
width =  raster_metadata['width']
mid_split_height = ceil(height * mid_split_size)

# windowing the original mid RGB file
north_window = Window(0, 0, width, mid_split_height)
city_window = Window(0, mid_split_height, width, mid_split_height)
south_window = Window(0, mid_split_height * 2, width, mid_split_height)

# re-opening the mid RGB file but this time with the windows specified
with rasterio.open('mid_RGB.tif') as mid_image:
    north_raster_data = mid_image.read(window=north_window)
    city_raster_data = mid_image.read(window=city_window)
    south_raster_data = mid_image.read(window=south_window)
    raster_data_profile = mid_image.profile.copy()

# creating profiles for each raster window
north_raster_profile = raster_data_profile.copy()
north_raster_profile.update({'height': north_window.height,
                             'width': north_window.width,
                             'transform': rasterio.windows.transform(north_window, raster_data_profile['transform'])})
    
city_raster_profile = raster_data_profile.copy()
city_raster_profile.update({'height': city_window.height,
                            'width': city_window.width,
                            'transform': rasterio.windows.transform(city_window, raster_data_profile['transform'])})
    
south_raster_profile = raster_data_profile.copy()
south_raster_profile.update({'height': south_window.height,
                             'width': south_window.width,
                             'transform': rasterio.windows.transform(south_window, raster_data_profile['transform'])})
   
# reading mid mask file with the windows specified
with rasterio.open('mid_mask.tif') as mid_mask:
    north_mask_data = mid_mask.read(window=north_window)
    city_mask_data = mid_mask.read(window=city_window)
    south_mask_data = mid_mask.read(window=south_window)
    mask_data_profile = mid_mask.profile.copy()


# creating profiles for each mask window
north_mask_profile = mask_data_profile.copy()
north_mask_profile.update({'height': north_window.height,
                           'width': north_window.width,
                           'transform': rasterio.windows.transform(north_window, mask_data_profile['transform'])})
    
city_mask_profile = mask_data_profile.copy()
city_mask_profile.update({'height': city_window.height,
                         'width': city_window.width,
                         'transform': rasterio.windows.transform(city_window, mask_data_profile['transform'])})
    
south_mask_profile = mask_data_profile.copy()
south_mask_profile.update({'height': south_window.height,
                           'width': south_window.width,
                           'transform': rasterio.windows.transform(south_window, mask_data_profile['transform'])})

# writing output images - raster
with rasterio.open('raw_images/midnorth_RGB.tif', 'w', **north_raster_profile) as dst:
        dst.write(north_raster_data)

with rasterio.open('raw_images/midcity_RGB.tif', 'w', **city_raster_profile) as dst:
        dst.write(city_raster_data)

with rasterio.open('raw_images/midsouth_RGB.tif', 'w', **south_raster_profile) as dst:
        dst.write(south_raster_data)
        
print("Mid RGB Image is succesfully split into North, City and South images with %s split" % mid_split_size)

# writing output images - mask
with rasterio.open('raw_images/midnorth_mask.tif', 'w', **north_mask_profile) as dst:
    dst.write(north_mask_data)
        
with rasterio.open('raw_images/midcity_mask.tif', 'w', **city_mask_profile) as dst:
    dst.write(city_mask_data)

with rasterio.open('raw_images/midsouth_mask.tif', 'w', **south_mask_profile) as dst:
        dst.write(south_mask_data)

print("Mid mask Image is succesfully split into North, City and South images with %s split" % mid_split_size)



#------------------------------------------------------------------------------
# 2. STEP
# Train-Test Split of the RBG Raster and the Mask files


# defining test size to be 20%
test_size = 0.2

for i in ['left','mid-north','mid-city','mid-south','right','island']: # complete split (all images)
"for i in ['midnorth','midcity','midsouth']:  # only mid image split
    
    # opening and reading the file    
    with rasterio.open("raw_images/" + i +'_RGB.tif') as src:
        raster_metadata = src.meta.copy()
        
    height = raster_metadata['height']
    width =  raster_metadata['width']
    
    # train_test split parameters for the three mid images vertically (option)
    """
    if "mid" in i: 
        test_width = width * test_size
        train_width = width - test_width
        
        train_window = Window(0, 0, train_width, height)
        test_window = Window(train_width, 0, test_width, height)
        
    # train_test split the other images horizontally        
    else:
    """ 
    
    # train_test split paramateres horizontally
    test_height = height * test_size
    train_height = height - test_height

    train_window = Window(0, 0, width, train_height)
    test_window = Window(0, train_height, width, test_height)
    
    # loading input raster image in split window
    with rasterio.open("raw_images/" + i +'_RGB.tif') as src:
        train_data = src.read(window=train_window)
        test_data = src.read(window=test_window)
        data_profile = src.profile.copy()

    # writing train data
    train_profile = data_profile.copy()
    train_profile.update({'height': train_window.height,
                          'width': train_window.width,
                          'transform': rasterio.windows.transform(train_window, data_profile['transform'])})

    with rasterio.open('train_test_split/'+ i + '_input_train.tif', 'w', **train_profile) as dst:
        dst.write(train_data)
  
      
    # writing test data
    test_profile = data_profile.copy()
    test_profile.update({'height': test_window.height,
                         'width': test_window.width,
                         'transform': rasterio.windows.transform(test_window, data_profile['transform'])})
    
    with rasterio.open('train_test_split/'+ i + '_input_test.tif', 'w', **test_profile) as dst:
        dst.write(test_data)
        
    print(i + "_input train-test split is ready")
    
    
    # loading mask image in split window
    with rasterio.open("raw_images/" + i +'_mask.tif') as src:
        train_data = src.read(window=train_window)
        test_data = src.read(window=test_window)
        data_profile = src.profile.copy()

    # writing train data
    train_profile = data_profile.copy()
    train_profile.update({'height': train_window.height,
                          'width': train_window.width,
                          'transform': rasterio.windows.transform(train_window, data_profile['transform'])})

    with rasterio.open('train_test_split/'+ i + '_mask_train.tif', 'w', **train_profile) as dst:
        dst.write(train_data)
      
    # writing test data
    test_profile = data_profile.copy()
    test_profile.update({'height': test_window.height,
                         'width': test_window.width,
                         'transform': rasterio.windows.transform(test_window, data_profile['transform'])})
    with rasterio.open('train_test_split/'+ i + '_mask_test.tif', 'w', **test_profile) as dst:
        dst.write(test_data)
    
    # tracking progress
    print(i + "_mask train-test split is ready")
