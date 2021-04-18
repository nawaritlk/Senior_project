import cv2 as cv
import matplotlib.pyplot as plt


def rescaleFrame(frame, scale = 0.75, width = 224, height = 224):
    if scale != 0:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
    dimension = (width,height)

    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


imgC = cv.imread(r'Test\ContentImg.jpg')
cv.imshow('Content Image',imgC)
# resized_imgC = rescaleFrame(imgC)
# cv.imshow('Content Image',resized_imgC)

imgS = cv.imread(r'StyleImage\Chakrabhan\0001.jpg')
print(type(imgS))
cv.imshow('Style Image',imgS)
# resized_imgS = rescaleFrame(imgS,0)
# cv.imshow('Style Image',resized_imgS)

grayC = cv.cvtColor(imgC,cv.COLOR_BGR2GRAY)
cv.imshow('Content Gray Image',grayC)

grayS = cv.cvtColor(imgS,cv.COLOR_BGR2GRAY)
cv.imshow('Style Gray Image',grayS)

# GrayScale Histogram
grayC_hist = cv.calcHist([grayC],[0],None,[256],[0,256])
plt.figure()
plt.title('GrayScale Content Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(grayC_hist)
plt.xlim([0,256])
plt.show()

grayS_hist = cv.calcHist([grayS],[0],None,[256],[0,256])
plt.figure()
plt.title('GrayScale Style Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(grayS_hist)
plt.xlim([0,256])
plt.show()


# histogram equalize
img = cv.imread(r'StyleImage/Chakrabhan/0001.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
dst = cv.equalizeHist(gray)
cv.imshow('Source Image',gray)
cv.imshow('Equalized Image',dst)
cv.waitKey(0)

# histogram matching
src = imgS
ref = imgC
from skimage import exposure
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel = multi)

cv.imshow('Source imgS',src)
cv.imshow('Ref imgC',ref)
cv.imshow('Match',matched)
cv.waitKey(0)