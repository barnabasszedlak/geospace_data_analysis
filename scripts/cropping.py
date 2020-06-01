# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:37:55 2020

@author: Szedlák Barnabás
"""

# imporing required libraries

import rasterio
import numpy as np


#------------------------------------------------------------------------------
# 1. STEP
# Reading the satellite and mask pictures
# raw input images from the raw_images folder


with rasterio.open('raw_images/WV2_2011-06-03_Left_RGB.tif') as src:
    #raster_left_data = src.read()
    raster_left_metadata = src.meta.copy()
    
with rasterio.open('raw_images/WV2_2011-06-03_Right_RGB.tif') as src:
    #raster_right_data = src.read()
    raster_right_metadata = src.meta.copy()
    
with rasterio.open('raw_images/WV2_2011-06-03_Mid_RGB.tif') as src:
    #raster_mid_data = src.read()
    raster_mid_metadata = src.meta.copy()
    
with rasterio.open('raw_images/WV2_2011-06-03_Island_RGB.tif') as src:
    #raster_island_data = src.read()
    raster_island_metadata = src.meta.copy()


# classification image

with rasterio.open('Classification.tif') as src:
    mask_data = src.read()
    mask_metadata = src.meta.copy()



#------------------------------------------------------------------------------
# 2. STEP
# cropping classification image to match raw input images


# defining pixel ranges based on Excel calculations
left_crop = {'y0':28692,'yn':40757,'x0':1341,'xn':6606}
mid_crop = {'y0':1277,'yn':70406,'x0':3753,'xn':68490}
right_crop = {'y0':1381,'yn':47509,'x0':31508,'xn':66141}
island_crop = {'y0':48674,'yn':69934,'x0':3497,'xn':43938}


# cropping mask_data pixels
mask_left = mask_data[0,left_crop['y0']:left_crop['y0'] + raster_left_metadata['height'],left_crop['x0']:left_crop['x0'] + raster_left_metadata['width']]
mask_left = np.expand_dims(mask_left, axis = 0)

mask_mid = mask_data[0,mid_crop['y0']:mid_crop['y0'] + raster_mid_metadata['height'],mid_crop['x0']:mid_crop['x0'] + raster_mid_metadata['width']]
mask_mid = np.expand_dims(mask_mid, axis = 0)

mask_right = mask_data[0,right_crop['y0']:right_crop['y0'] + raster_right_metadata['height'],right_crop['x0']:right_crop['x0'] + raster_right_metadata['width']]
mask_right = np.expand_dims(mask_right, axis = 0)

mask_island = mask_data[0,island_crop['y0']:island_crop['y0'] + raster_island_metadata['height'],island_crop['x0']:island_crop['x0']+ raster_island_metadata['width']]
mask_island = np.expand_dims(mask_island, axis = 0)


# creating mask files with updated metadata
maskaff = mask_metadata['transform']
for i in ['left','mid','right','island']:
    
    vars()["mask_"+ i +"_metadata"] = mask_metadata.copy()
    vars()["mask_"+ i +"_metadata"].update({'height': vars()["raster_"+ i +"_metadata"]['height'],'width':vars()["raster_"+ i +"_metadata"]['width']})

    rasteraff = vars()["raster_"+ i +"_metadata"]['transform']
    newaff = rasterio.Affine(maskaff[0], maskaff[1], rasteraff[2], maskaff[3], maskaff[4], rasteraff[5])
    vars()["mask_"+ i +"_metadata"].update({'transform': newaff})


    with rasterio.open(i+"_mask.tif", 'w', **vars()["mask_"+ i +"_metadata"]) as dst:
        dst.write(vars()['mask_'+i])
    print(i + "_mask.tif is saved.")
    
