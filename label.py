import cv2
import numpy as np
import os

def warpImg(img,src_points,tar_points,size):
    h,s = cv2.findHomography(src_points,tar_points)
    imghomo = cv2.warpPerspective(img,h,size)
    return imghomo

def getshape(points):
    w = max(points[:,0]) - min(points[:,0])
    h = max(points[:,1]) - min(points[:,1])
    return np.array([0,0,w,0,w,h,0,h],np.int32).reshape(-1,2),(h,w)

#click event function
def click_event(event, x, y, flags, param):
    global points,img
    draw = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])
        print(x,"/",y)
        cv2.circle(draw, (x,y), 5 ,(255,0,0), -1)
        cv2.imshow("img", draw)


#Here, you need to change the image name and it's path according to your directory
imgdir = "rename_total_images_train/"
txtdir = "rename_total_images_train_labels/"
outdir = "extraCrop/"
# files = os.listdir(imgdir)
txtfile = "img_894.txt"

with open(txtdir+txtfile,'w',encoding = "utf-8") as f:
    # for fname in files:
    points = []
    img = cv2.imread(imgdir+'img_894.jpg')
    # img = cv2.resize(img,(600,800))
    cv2.imshow("img",img)
    #calling the mouse click event
    cv2.setMouseCallback("img", click_event)
    cv2.waitKey(0)        
    src = np.array(points.copy(),np.int32).reshape(-1)
    src = [str(i) for i in src]
    f.write(','.join(src)+",1")
        # warpedImgs = []
        # if len(src) == 0:
        #     continue
        # for s in src:
        #     tar,size = getshape(s)
        #     warped = warpImg(img,s,tar,(size[1],size[0]))
        #     warpedImgs.append(warped)

        # index = 1
        # for warped in warpedImgs:
        #     cv2.imwrite(outdir+fname[:-3]+"_"+str(index)+".jpg",warped)
        #     f.write(outdir+fname[:-3]+"_"+str(index)+".jpg\t\n")
        #     index += 1
cv2.destroyAllWindows()


