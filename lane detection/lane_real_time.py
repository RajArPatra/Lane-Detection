# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:23:01 2020

@author: Raj Ar
"""


import cv2

import numpy as np


'''define the path of the video/image'''
cap = cv2.VideoCapture("t.mp4")

while(True):
    
    qret, img44 = cap.read()
    
    if qret==1:
        
        '''Define the region of Intrest'''
        img1=img44[130:,:]
        img2=np.zeros(img44.shape)
        img2[130:,:]=img1
        img2=img2.astype('uint8')
        
        
        '''Thresold the image'''
        higher=np.array([244,244,244])    
        lower=np.array([134,142,45])
        mask3=cv2.inRange(img2,lower,higher)
        color=cv2.bitwise_and(img2,img2,mask=mask3) 
    
        '''define the matrix for src,dst for perspective transform'''
        src = np.array([[450,80],
                    [615, 80],
                    [3, 544],
                    [1276, 544]], dtype=np.float32)
        dst=np.array([[3, 80],
                      [1276, 80],
                      [3, 544],
                      [1276,544]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src, dst)
        iM = cv2.getPerspectiveTransform(dst, src)
        
        img7=cv2.warpPerspective(color,M,(color.shape[1],color.shape[0]),
                                         flags=cv2.INTER_LINEAR)
        trans=cv2.warpPerspective(img44,M,(img44.shape[1],img44.shape[0]),
                                         flags=cv2.INTER_LINEAR)
        
        img3=img7 
        img3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
        img=img3.copy()
        histogram = np.sum(img3[img3.shape[0]//2:,:], axis=0)
        mid = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:mid])
        rightx_base = np.argmax(histogram[mid:]) + mid
        nwindows=100
        # Set height of windows
        '''deffine the height of each window'''
        window_height = np.int((550-250) // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        mid_left=[]
        mid_right=[]
        minpix=1
        margin=10
        draw_windows = True
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            '''replace 550 with  height of the image'''
            win_y_low = 550 - (window + 1) * window_height
            win_y_high = 550 - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            mid_xleft= (win_xleft_low+win_xleft_high)//2
            mid_xright=(win_xright_low+win_xright_high)//2
            mid_y=(win_y_low+win_y_high)//2
            mid_left.append(( mid_xleft,mid_y))
            mid_right.append((mid_xright,mid_y))
            
            # Draw the windows on the visualization image
            if draw_windows == True:
                cv2.rectangle(img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                              (100, 255, 255),2 )
                cv2.rectangle(img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                              (100, 255, 255), 2)
                # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
       
        mid_right.reverse()
        points=[*mid_left,*mid_right]
        points=np.array(points)
        print(points)
        cv2.fillConvexPoly(trans, points, (242, 0,0), lineType=8, shift=0)
       
        img1=cv2.warpPerspective(trans,iM,(trans.shape[1],trans.shape[0]),
                                         flags=cv2.INTER_LINEAR)
        
        
        '''points coordinate array'''
        points_img=np.zeros_like(cv2.cvtColor(trans,cv2.COLOR_BGR2GRAY))
        '''replace 1280 with width of image and 720 with height of image'''
        if(points[:,0].all()<1280 & points[:,1].all()<720):
         points_img[points]=255
        points_img_real=cv2.warpPerspective(points_img,iM,(points_img.shape[1],points_img.shape[0]),
                                         flags=cv2.INTER_LINEAR)
      
        
        
        higher=np.array([244,244,244])    
        lower=np.array([242,0,0])
            
        mask2=cv2.inRange(img1,lower,higher)
        
    
        color1=cv2.bitwise_and(img1,img1,mask=mask2) 
        color1=cv2.cvtColor(color1,cv2.COLOR_BGR2GRAY)
    
        
        cnts,_=cv2.findContours(color1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img44,cnts,-1,(255,255,0),10)              
        
        
        cv2.imshow("out",img44)
        if cv2.waitKey(24)==27:        
            cv2.destroyAllWindows()
            break
    else:
        break
    
    
cap.release()
cv2.destroyAllWindows()
   
    