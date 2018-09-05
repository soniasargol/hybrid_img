import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage



def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))

        # downsample image
        cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # pad the top to append to the output
        pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))

    return output

def im2single(im):
    # print("image:",im)
    im = im.astype(np.float32) / 255
    return im

def single2im(im):
    im *= 255
    im = im.astype(np.uint8)
    return im

def load_image(path):
    return im2single(cv2.imread(path))[:, :, ::-1]

def save_image(path, im):
    return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])

def gaussian_filter(i, j, x, y, sigma):
    return np.exp(-((i-((x-1)/2.))**2 + (j-((y-1)/2.))**2)/(2.*sigma**2))



def scaleSpectrum(A):
   return np.real(np.log10(np.absolute(A) + np.ones(A.shape)))



def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
    centerI = int(numRows/2) + 1
    centerJ = int(numCols/2) + 1

    def gaussian(i,j):
        coefficient = np.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
        # if i%2==0:
        #     print("coefficient:",coefficient)
        # print("coefficient:",1 - coefficient if highPass else coefficient)
        return 1 - coefficient if highPass else coefficient
    # print(np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)]))
    return np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])


def filterDFT(imageMatrix, filterMatrix):
    print("filterMatrix:",filterMatrix)
    shiftedDFT = fftshift(fft2(imageMatrix[:, :, 0]))
    misc.imsave("dft.png", scaleSpectrum(shiftedDFT))

    filteredDFT = shiftedDFT * filterMatrix
    misc.imsave("filtered-dft.png", scaleSpectrum(filteredDFT))
    return ifft2(ifftshift(filteredDFT))


def lowPass(imageMatrix, sigma):
    n,m,l = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))


def highPass(imageMatrix, sigma):
    print(imageMatrix.shape)
    n,m,l = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))


def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
    highPassed = highPass(highFreqImg, sigmaHigh)
    lowPassed = lowPass(lowFreqImg, sigmaLow)

    return highPassed + lowPassed
