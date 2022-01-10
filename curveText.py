import cv2
import numpy as np
from util import *
import matplotlib.pyplot as plt

rawDir = 'test_images/'
predDir = 'pred/'
fname = '1004'
raw_img = cv2.imread(rawDir + fname + ".jpg")
h,w = raw_img.shape[:2]
point_size = min(h,w) // 100
img = cv2.imread(predDir + fname + ".jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (w,h))
# img = cv2.GaussianBlur(img,(9,9),0)
binarize = np.array(img > 100,dtype = np.uint8)*255
cv2.imwrite(fname + 'binarize.jpg',binarize)
cnts, _ = cv2.findContours(binarize.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contour = [c for c in cnts if len(c) > 10]
for cIdx,c in enumerate(contour):
    c = np.array(c).reshape(-1,2)
    expandcontour = unclip(c,unclip_ratio = 3).reshape(-1,2)
    expand = cv2.approxPolyDP(expandcontour, cv2.arcLength(expandcontour,True) * 0.02, True).reshape(-1,2)
    # cv2.drawContours(raw_img, [expand], 0, (255,0,0), point_size//4)
    side_idx = side_detect(expand)
    if side_idx is not None:
        cv2.line(raw_img,expand[side_idx[0]],expand[side_idx[1]],(255,100,100),point_size//2)
        cv2.line(raw_img,expand[side_idx[2]],expand[side_idx[3]],(255,100,100),point_size//2)
        sides = get_2_side(expandcontour,[tuple(expand[i]) for i in side_idx])
        upper_side = np.array(sides[0],dtype=np.int32).reshape(-1,2)
        # print(upper_side.shape)
        control_point = bezier_fit(upper_side[:,0],upper_side[:,1])
        # for cp in upper_side:
            # cv2.circle(raw_img,(int(cp[0]),int(cp[1])),point_size,(255,0,255),-1)
        curveX,curveY = bezier_curve(control_point)
        for x,y in zip(curveX,curveY):
            cv2.circle(raw_img,(int(x),int(y)),point_size//4,(255,55,255),-1)

        bottom_side = np.array(sides[1],dtype=np.int32).reshape(-1,2)
        # print(bottom_side.shape)
        control_point = bezier_fit(bottom_side[:,0],bottom_side[:,1])
        # for cp in control_point:
        #     cv2.circle(raw_img,(int(cp[0]),int(cp[1])),point_size,(255,255,0),-1)
        curveX,curveY = bezier_curve(control_point)
        for x,y in zip(curveX,curveY):
            cv2.circle(raw_img,(int(x),int(y)),point_size//4,(255,255,55),-1)
    else :
        box = cv2.minAreaRect(expandcontour)
        box = np.int0(cv2.boxPoints(box))
        cv2.drawContours(raw_img, [box], -1, (0, 0, 0), point_size//4)
# plt.imshow(raw_img)
# plt.show()
cv2.imwrite(fname + 'curve.jpg',raw_img)
