'''
原理参考：https://blog.csdn.net/carson2005/article/details/7581642
代码参考：https://blog.csdn.net/wsp_1138886114/article/details/84400392
'''

import numpy as np
import cv2

cap = cv2.VideoCapture('./imgs/pedestrian.mp4')

# 角点检测的参数：角点最大数量，品质因子，角点间的最小距离，
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# 光流法参数
# maxLevel 未使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机生成的颜色
color = np.random.randint(0, 255, (100, 3))


ret, old_frame = cap.read()                             # 取出视频的第一帧
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # 灰度化
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # 对第一帧图像进行角点检测
mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片

while True:
    _, frame = cap.read()  # 循环读入每一帧
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算光流以获取点的新位置
    '''
    输入：
    previmage:前一帧图像
    nextimage:当前帧图像
    prevPts:待跟踪的特征向量
    winSize:搜索窗口大小3*3/5*5....
    maxLevel:最大金字塔层数
    返回：
    nextPts:输出跟踪特征点向量
    status:是否找到特征点
    '''
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # status为1的角点
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # 绘制跟踪框
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30)  # & 0xff
    if k == 27:
        break
    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
