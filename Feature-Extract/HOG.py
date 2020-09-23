'''
原理和代码参考
https://zhuanlan.zhihu.com/p/85829145
https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
'''

import cv2
import numpy as np

# 0表示将图片以灰度读出来
img = cv2.imread('./imgs/umbrella.jpg', 0)
# Gamma校正
img = np.power(img/float(np.max(img)), 1.5)
'''
cv2.imshow('After Gamma', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# 计算水平方向和垂直方向的梯度，以及梯度幅值图和方向图
# cv2.Sobel()的详解：https://blog.csdn.net/qq_27261889/article/details/80891491
img = np.float32(img)
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
img, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
'''
cv2.imshow('GX', gx)
cv2.imshow('GY', gy)
cv2.imshow('AMP', img)
cv2.imshow('ANG', angle)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

from skimage import feature, exposure
image = cv2.imread('./imgs/tst1.jpg')

# feature.hog详解：https://www.jianshu.com/p/d3f93c360226
fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
print(len(fd))   # 9720  原图片是268*304  9bins * 2*2 * （[268/16]-1）*（[304/16]-1）
# 调整强度，exposure.rescale_intensity详解：https://blog.csdn.net/haoji007/article/details/52063252
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

cv2.imshow('img', image)
cv2.imshow('hog', hog_image_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

