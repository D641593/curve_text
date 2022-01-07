import cv2
import numpy as np
from util import *

raw_img = cv2.imread("img_6227_raw.jpg")
h,w = raw_img.shape[:2]
img = cv2.imread("img_6227_hand.jpg")
img = cv2.resize(img, (w,h))
grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
binarize = np.array(grey > 100,dtype = np.uint8)*255
binarize = cv2.GaussianBlur(binarize,(9,9),0)

cnts, _ = cv2.findContours(binarize.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = np.array(cnts)
contourLength = [cv2.arcLength(cnt,True) for cnt in cnts]
contourLength, contour = zip(*sorted(zip(contourLength,contour),reverse=True))
for i, c in enumerate(contour):
    if contourLength[i] < 100:
        break
    (x,y,w,h) = cv2.boundingRect(c)
    sub_img = binarize[y:y+h,x:x+w]
    # cv2.imwrite('subimg_binarize_6227_hand.jpg',sub_img)
    # cv2.imwrite('subimg_6227_hand.jpg',raw_img[y:y+h,x:x+w])
    sub_img = cv2.GaussianBlur(sub_img,(5,5),0)
    thinned = cv2.ximgproc.thinning(sub_img)
    # cv2.imwrite('subimg_thin_6227_hand.jpg',thinned)
    edges = findedge(thinned)
    edges = sorted(edges,key=len,reverse=True)
    edge = edges[0]
    side_1,side_2 = get_side(edge,sub_img)
    # cv2.line(thinned,(side_1[0][1],side_1[0][0]),(side_1[1][1],side_1[1][0]),(255,255,0),1)
    # cv2.line(thinned,(side_2[0][1],side_2[0][0]),(side_2[1][1],side_2[1][0]),(255,0,255),1)
    # cv2.imwrite('subimg_side_6227_hand.jpg',thinned)
    contour = np.array(c)
    contour = contour.squeeze(1)
    contour = contour - (x,y)
    point_1 = find_close_point(contour,side_1[0])
    point_2 = find_close_point(contour,side_1[1])
    point_3 = find_close_point(contour,side_2[0])
    point_4 = find_close_point(contour,side_2[1])
    sides = get_2_side(contour,[point_1,point_2,point_3,point_4])
    if sides == None:
        break

    points = []
    raw_sub = raw_img[y:y+h,x:x+w]
    upper_side = sides[0]
    step = (len(upper_side)-1) / 4
    for idx in range(5):
        cv2.circle(raw_sub,upper_side[round(step*idx)],5,(255,255,0),-1)
        points.append(str(upper_side[round(step*idx)][0]))
        points.append(str(upper_side[round(step*idx)][1]))

    bottom_side = sides[1]
    step = (len(bottom_side)-1) / 4
    for idx in range(5):
        cv2.circle(raw_sub,bottom_side[round(step*idx)],5,(255,0,255),-1)
        points.append(str(bottom_side[round(step*idx)][0]))
        points.append(str(bottom_side[round(step*idx)][1]))

    with open('subimg_6227_hand.txt','w',encoding='utf-8') as wf:
        wf.write(','.join(points))
        wf.write(",label")
pip install opencv-python