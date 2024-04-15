"""
Copyright 2021, Zhu Aoyu, CUG.
Depth map visualization with Adaptive Contrast Enhancement.
"""
import numpy as np
from struct import *
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
import cv2
from glob import  glob


def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]
        
        array = np.fromfile(fid, np.float32)
    array = array.reshape((channel, width, height), order="F")
    return np.transpose(array, (2, 1, 0)).squeeze()

def read_gipuma_dmb_int(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.int32)
    array = array.reshape((channel, width, height), order="F")
    return np.transpose(array, (2, 1, 0)).squeeze()

def normal_360(path_input,save_path,count):
    path_input = os.path.join(path_input,"{}.dmb".format(os.path.basename(save_path)))
    save_path = os.path.join(save_path,'{0:0>2}.jpg'.format(count))
    normal_image = read_gipuma_dmb(path_input)
    div = np.ones_like(normal_image[:,:,0]).astype('float')
    div = normal_image[:,:,0]*normal_image[:,:,0]+normal_image[:,:,1]*normal_image[:,:,1]+normal_image[:,:,2]*normal_image[:,:,2]
    mask_div = (div == 0)
    div[mask_div] = 1
    div = np.sqrt(div)
    div = np.expand_dims(div,axis=2)
    div_3 = np.concatenate((div,div,div),axis=2)
    normal_image = normal_image/div_3
    normal_image = (normal_image )*127 + 127
    normal_image = Image.fromarray(normal_image.astype('uint8'))
    normal_image.save(save_path)

def cost(path_input,save_path,count):
    path_input = os.path.join(path_input,"costs.dmb")
    save_path = os.path.join(save_path,'{0:0>2}.jpg'.format(count))
    cost_image = read_gipuma_dmb(path_input)
    upbound = np.nanmax(cost_image)
    cost_image[np.isnan(cost_image)] = upbound
    cost_image = cost_image/3*255
    cost_image = Image.fromarray(cost_image.astype('uint8'))
    cost_image.save(save_path)

def depth(path_input,save_path,depth_range,count):
    path_input = os.path.join(path_input,"{}.dmb".format(os.path.basename(save_path)))
    if not os.path.exists(path_input):
        print("{path_input} dosen't exsits.".format(path_input))
        return
    save_path = os.path.join(save_path, '{0:0>2}.jpg'.format(count))
    depth_image = read_gipuma_dmb(path_input)
    depth_image[np.isnan(depth_image)] = 0
    upbound = np.nanmax(depth_image)
    lowbound = np.nanmin(depth_image)
    # [h,w] = depth_image.shape
    mask2 = (depth_image < depth_range[0])
    depth_image[mask2] = 0
    mask3 = (depth_image > depth_range[1])
    depth_image[mask3] = depth_range[1]
    mask4 = mask2 + mask3
    mask4 = (mask4==0)
    avg  = np.mean(depth_image[mask4])
    std = np.std(depth_image[mask4])
    var = np.var(depth_image[mask4])
    depth_image2 = (((depth_image - avg) / std)*0.333)
    depth_image3 = ((((depth_image - avg) / std)*128)+127).astype('uint8')
    Mysigmoid = 1/(np.exp((-8*depth_image2))+1) * 255
    Mysigmoid = Mysigmoid.astype('uint8')
    plt.figure()
    plt.imsave(save_path,Mysigmoid,cmap="jet")
    plt.close()

def View_select(path_input,save_path,count):
    path_input = os.path.join(path_input,"{}.dmb".format(os.path.basename(save_path)))
    if not os.path.exists(path_input):
        print("{path_input} dosen't exsits.".format(path_input))
        return
    depth_image = read_gipuma_dmb_int(path_input)
    depth_image[np.isnan(depth_image)] = 0
    save_path = os.path.join(save_path,'{0:0>2}.jpg'.format(count))
    plt.figure()
    plt.imsave(save_path,depth_image,cmap="jet")
    plt.close()

def read_depth_range(path):
    cam_path = os.path.join(os.path.dirname(path),"cams")
    cams_abs = [os.path.join(cam_path,i) for i in os.listdir(cam_path)]
    depth_ranges = np.zeros([len(cams_abs),2]).astype(np.float64)
    count = 0 
    for cam_abs in cams_abs:
        with open(cam_abs, "r") as fid:
            content = fid.readlines()
            depth_ranges[count,0] = content[-1].split(' ')[0]
            depth_ranges[count,1] = content[-1].split(' ')[-1]
            depth_ranges[count,0] = depth_ranges[count,0].astype(np.float64) * 0.6
            depth_ranges[count,1] = depth_ranges[count,1].astype(np.float64) * 1.2
        count = count+1
    return depth_ranges

if __name__ == '__main__':
    count = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default = r"F:\ZAY\datasets_beta\ETH3D\tree_earth\CNVR")
    args = parser.parse_args()
    if(args.path):
        path = args.path
    sub_dirs = [t for t in os.listdir(path) if '2333' in  t]
    # normal cost depth maps
    save_root = os.path.join(path,"NCD")
    depth_ranges =  read_depth_range(path)
    for i,sub_dir in enumerate(sub_dirs) :
        files = os.listdir(os.path.join(path,sub_dir))
        for file in files:
            save_path = os.path.join(save_root,file.split(".")[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            depth_range = depth_ranges[i]
            path_input = os.path.join(path,sub_dir)
            if "cost" in file:
                cost(path_input,save_path,i)
            if "depths" in file:
                depth(path_input,save_path,depth_range,i)
            if "normal" in file:
                normal_360(path_input,save_path,i)
            if "view" in file:
                    View_select(path_input,save_path,i)
