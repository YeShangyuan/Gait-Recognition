#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:12:26 2017

@author: wwb
"""

import os
import os.path
import numpy as np
import random
import cv2
from PIL import Image
from scipy.misc import imsave 

def lunkuo(image):
    rowImg,colImg = image.shape
    up = -1
    buttom = -1
    left = -1
    right = -1
    for row in range(0,rowImg):
        for col in range(0,colImg):
            if image[row,col] >= 250:
                up = row
                break
        if up != -1:
            break


    for row in range(0,rowImg):
        for col in range(0,colImg):
            if image[rowImg-1-row,col] >= 250:
                buttom = rowImg-1-row
                break
        if buttom != -1:
            break

    for col in range(0,colImg):
        for row in range(0,rowImg):
            if image[row, col] >= 250:
                left = col
                break
        if left != -1:
            break

    for col in range(0,colImg):
        for row in range(0,rowImg):
            if image[row, colImg-1-col] >= 250:
                right =colImg-1-col
                break
        if right != -1:
            break
    imageTemp = image[up:buttom,left:right]
    r, c = imageTemp.shape
    if (r-c)%2 == 0:
        imageTemp = np.column_stack([np.uint8(np.zeros([r,(r-c)/2])),np.uint8(imageTemp),np.uint8(np.zeros([r,(r-c)/2]))])
    else:
        imageTemp = np.column_stack([np.uint8(np.zeros([r,(r-c+1)/2])),np.uint8(imageTemp),np.uint8(np.zeros([r,(r-c+1)/2]))])
    
    d = cv2.resize(imageTemp, (96,96), interpolation = cv2.INTER_AREA)
# =============================================================================
#     rows,cols=d.shape
#     for i in range(rows):
#         for j in range(cols):
#             if (d[i,j]<=128):
#                 d[i,j]=0
#             else:
#                 d[i,j]=1
# =============================================================================
    return d
                
                

solution = 96*96
labelNum = 3
rootDir = '/home/wwb/Documents/programes/gait_cnn/temp/data_sk2_3_80'
dirList = os.listdir(rootDir)
ImagesTemp = np.zeros([0,solution])
LabelsTemp = np.zeros([0,1])
imageCount = 0
for i in range(0, len(dirList)):
    personPath = os.path.join(rootDir, dirList[i])
    imageList = os.listdir(personPath)
    for xImage in range(0,len(imageList)):
        xImagePath = os.path.join(personPath, imageList[xImage])
        image = cv2.imread(xImagePath,cv2.COLOR_BGR2GRAY)
        cutImage = lunkuo(image)
        filename = personPath + '/' + str(xImage)+'.png'
        imsave(filename,cutImage)
        os.remove(xImagePath)
        imageCount = imageCount + 1
        row, col= cutImage.shape
        ImagesTemp = np.row_stack((ImagesTemp,cutImage.reshape([1,row*col])))
        LabelsTemp = np.row_stack((LabelsTemp,i))
        
# =============================================================================
# trainImages = np.zeros([ImagesTemp.shape[0],solution])
# trainLabels = np.zeros([LabelsTemp.shape[0],labelNum])
# 
# trainLabelsTempNew = np.zeros([ImagesTemp.shape[0],labelNum])
# 
# for i in range(0,len(dirList)):
#     trainLabelsTempNew[np.where(LabelsTemp == i+1)[0],i] = 1
#     
# # =============================================================================
# # testImages = (testImagesTemp) / 255.0
# # testLabels = testLabelsTempNew
# # =============================================================================
# 
# L = range(0,ImagesTemp.shape[0])
# random.shuffle(L)
# for i in range(0, len(L)):
#      trainImages[i,:] = (ImagesTemp[L[i],:]) / 255.0
#      trainLabels[i,:] = trainLabelsTempNew[L[i],:]
# =============================================================================

np.save("/home/wwb/Documents/programes/gait_cnn/temp/Images.npy", ImagesTemp)
np.save("/home/wwb/Documents/programes/gait_cnn/temp/Labels.npy", LabelsTemp)
