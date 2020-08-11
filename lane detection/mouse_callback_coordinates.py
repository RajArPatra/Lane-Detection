# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:42:57 2020

@author: HP
"""

import pickle
import cv2
import os
import numpy as np



global img
'''define the path of the image to detect the coordinte for transform'''
img=cv2.imread("e.jpg")
window_name = 'Image'

def call_func(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        center = (x,y)
        org = (x, y)
        print (x)
        print (y)
        global img
        font = cv2.FONT_HERSHEY_SIMPLEX

        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        #img = cv2.putText(img, x , org, font,  
                   #fontScale, color, thickness, cv2.LINE_AA)
             
        cv2.circle(img, center, 2, (255, 0, 0), 2)
cv2.setMouseCallback('Image',call_func)
while True:
    
    cv2.setMouseCallback('Image',call_func,)
    
    cv2.imshow('Image', img)    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
