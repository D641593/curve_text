import cv2
import numpy as np
from numpy.core.numeric import full
from scipy.stats import linregress
from shapely.geometry import Polygon
import pyclipper
import math
from numpy import array, linalg, matrix
from scipy.special import comb

Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*comb(n,k)
bezierM = lambda ts: matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])

def bezier_fit(x, y):
    if len(x) < 4:
        print('here')
        return np.array([[x[0],y[0]],[x[0]*2//3 + x[1] // 3,y[0]*2//3 + y[1] // 3],[x[0]//3 + x[1]*2 // 3,y[0]//3 + y[1]*2 // 3],[x[1],y[1]]])
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))
    Pseudoinverse = np.linalg.pinv(bezierM(t)) 
    control_points = Pseudoinverse.dot(data)
    return np.asarray(control_points).reshape(-1,2)

def bezier_curve(points, nTimes=1000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ Mtk(nPoints-1, t, i) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def findedge(img): 
    edges = [] # 用來記錄所有edge，也是函式輸出
    h,w = img.shape[:2]
    label = np.zeros((h,w)) # 一個全是0的矩陣，跟圖片一樣大小，用來記錄該點是否有被搜尋過
    dirs = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),] # 尋找方向
    for i in range(h):
        for j in range(w): # 讀長寬搜尋edge
            if label[(i,j)] != 0 or img[(i,j)] == 0: # 如果這點已經搜尋過或者不是edge，跳過
                continue
            point = (i,j) # 找到edge
            flag = 0
            for dir in dirs: # 檢查是否為端點
                y = point[0] + dir[0]
                x = point[1] + dir[1] # 找點的各個方向
                if y < 0 or y >= h or x < 0 or x >= w: # 超出圖片不找
                    continue
                if label[(y,x)] != 0 or img[(y,x)] == 0: # 如果這點已經搜尋過或者不是edge，跳過
                    continue
                flag += 1
                if flag > 1: # 不是端點，不從此點開始
                    break
            if flag > 1:
                continue
            label[point] = 1 # 把這點標記為搜尋過
            tmp = [point] # 用來記錄跟這點相鄰的edge
            num = 0 # 紀錄搜尋點的次數
            while True:
                for dir in dirs:
                    y = point[0] + dir[0]
                    x = point[1] + dir[1] # 找點的各個方向
                    if y < 0 or y >= h or x < 0 or x >= w: # 超出圖片不找
                        continue
                    if label[(y,x)] != 0 or img[(y,x)] == 0: # 如果這點已經搜尋過或者不是edge，跳過
                        continue
                    tmp.append((y,x)) # 紀錄起來
                    label[(y,x)] = 1 # 標記以搜尋過
                num += 1 # 搜尋完一個點的八方向就+1
                if num == len(tmp): # 如果tmp裡的點都被搜尋過了，就跳出迴圈，這一類edge搜尋完成
                    break
                point = tmp[num] # tmp還沒搜尋完，將point設成tmp裡的下一個點
            edges.append(tmp) # 搜尋完成後塞進去
    return edges # 找完整張圖，回傳所有edge

def get_side(edge,img,sample_point_num = 10):
    # edge > [(y,x),...]
    # img > binarize img before thinning
    # sample_point_num > number of points to fit line
    points = np.array(edge[:sample_point_num])
    lr = linregress(points[:,1],points[:,0])
    if lr.slope < 0.005 and lr.slope > -0.005:
        slope = -1
    else:
        slope = -1 / lr.slope
    side_1 = []
    side_1.append(get_side_point(points[len(points)//2],img,5,slope,lr))
    side_1.append(get_side_point(points[len(points)//2],img,-5,slope,lr))

    points = np.array(edge[-sample_point_num:])
    lr = linregress(points[:,1],points[:,0])
    if lr.slope < 0.005 and lr.slope > -0.005:
        slope = -1
    else:
        slope = -1 / lr.slope
    side_2 = []
    side_2.append(get_side_point(points[len(points)//2],img,5,slope,lr))
    side_2.append(get_side_point(points[len(points)//2],img,-5,slope,lr))
    return side_1,side_2
    
def get_side_point(end_point, img, step, slope, lr):
    side_x = end_point[1]
    side_y = end_point[0]
    side_offset = step
    h,w = img.shape[:2]
    while True:
        tmp_x = side_x + side_offset
        tmp_y = round(side_y + side_offset * slope)
        if tmp_x >= 0 and tmp_x < w and tmp_y >= 0 and tmp_y < h and img[tmp_y,tmp_x] != 0:
            side_offset += step
        else:
            low_step = -1 * step // abs(step)
            for i in range(abs(step)):
                target_x = tmp_x + low_step * i
                target_y = round(tmp_y + low_step * i * slope)
                if target_x >= 0 and target_x < w and target_y >= 0 and target_y < h and img[target_y,target_x] != 0:
                    break
            break
    return (target_y,target_x)

def find_close_point(contour,point):
    # contour > [[x,y],...]
    # point > [y,x]
    dis = None
    tx,ty = 0,0
    for p in contour:
        x = p[0] - point[1]
        y = p[1] - point[0]
        if dis == None:
            dis = x**2 + y**2
            tx = p[0]
            ty = p[1]
        elif (x**2 + y**2) < dis:
            dis = x**2 + y**2
            tx = p[0]
            ty = p[1]
    return (tx,ty)

def get_2_side(contour, side_points):
    assert len(side_points) == 4,'side_points must have 4 point.'
    sides = []
    tmp = []
    for c in contour:
        c = tuple(c)
        tmp.append(c)
        if c in side_points:
            sides.append(tmp.copy())
            tmp.clear()
            tmp.append(c)
    sides.append(tmp)
    if len(sides) < 4:
        return None
    if len(sides) == 5:
        sides[4].extend(sides[0])
        sides[0] = sides[4][::-1]

    target = []
    for i in range(4):
        if side_points[0] in sides[i] and side_points[1] in sides[i]:
            continue
        elif side_points[2] in sides[i] and side_points[3] in sides[i]:
            continue
        else:
            target.append(sides[i])
    target[1] = target[1][::-1]
    return target


def side_detect(contour):
    contourLength = len(contour)
    if contourLength < 4:
        return None
    idx = list(range(contourLength))
    der1 = [contour[i]-contour[i-1] for i in idx]
    der1 = [der1[i]/math.sqrt(der1[i][1]**2 + der1[i][0]**2) for i in idx]
    der2 = [der1[i+1]-der1[i] for i in idx[:-1]]
    der2.append(der1[0]-der1[-1])
    score = [der2[i][0]**2+der2[i][1]**2 for i in idx]
    score, idx = zip(*sorted(zip(score,idx),reverse=True))
    pointsIdx = np.array(idx[:4])
    
    pair = []
    for i in range(4):
        pair.append(side_pair(contour,pointsIdx,i))
    if pair[0][0] != pair[pair[0][1]][1] or pair[1][0] != pair[pair[1][1]][1] or pair[2][0] != pair[pair[2][1]][1] or pair[3][0] != pair[pair[3][1]][1]:
        return None
    else:
        order = pair[0].copy()
        order.extend([i for i in range(4) if i not in pair[0]])
        return pointsIdx[order]

def side_pair(contour,pointsIdx,idx):
    mindis = None
    side1 = [idx]
    option = [i for i in range(4) if i != idx]
    full_distance = count_distance(contour,0,len(contour))
    for i in option:
        dis = count_distance(contour,pointsIdx[idx],pointsIdx[i],full_distance=full_distance)
        if mindis is None or mindis > dis:
            mindis = dis
            minIdx = i
    side1.append(minIdx)
    return side1

def count_distance(contour,start,end, full_distance=None):
    contourLength = len(contour)
    if start > end:
        tmp = start 
        start = end
        end = tmp
    dis1 = 0
    dis2 = 0
    if full_distance is not None:
        for i in range(start,end):
            dis1 += math.sqrt((contour[i][0] - contour[i+1][0])**2 + (contour[i][1] - contour[i+1][1])**2)
        return min(dis1,full_distance-dis1)
    else:
        for i in range(0,start):
            dis1 += math.sqrt((contour[i][0] - contour[(i+1)%contourLength][0])**2 + (contour[i][1] - contour[(i+1)%contourLength][1])**2)
        for i in range(start,end):
            dis2 += math.sqrt((contour[i][0] - contour[(i+1)%contourLength][0])**2 + (contour[i][1] - contour[(i+1)%contourLength][1])**2)
        for i in range(end,contourLength):
            dis1 += math.sqrt((contour[i][0] - contour[(i+1)%contourLength][0])**2 + (contour[i][1] - contour[(i+1)%contourLength][1])**2)
        if dis1 == 0:
            return dis2
        elif dis2 == 0:
            return dis1
        else:
            return min(dis1,dis2)

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.PT_CLIP, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded
    

