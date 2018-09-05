from student_code import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #added
from scipy import misc
import numpy as np
# #im = Image.imread("cat.bmp")
# im2=load_image("einstein.bmp")
# im1=load_image("marilyn.bmp")
# # k, l = input("Filter dim:").split()
# k, l= 19,19
# f = np.zeros((k, l))
# sigma = 2
# # count=0
# for i in range(0, k):
#     for j in range(0, l):
#         # f[i, j] = gaussian_filter(i, j, k, l, sigma)*((-1)**count)
#         f[i, j] = gaussian_filter(i, j, k, l, sigma)
#         # count +=1
# f = f/np.sum(f)
#
# f_h = np.zeros((k, l))
# sigma2 = 1.7
# for i in range(0, k):
#     for j in range(0, l):
#         # f[i, j] = gaussian_filter(i, j, k, l, sigma)*((-1)**count)
#         f_h[i, j] = gaussian_filter(i, j, k, l, sigma2)
#         # count +=1
# f_h = f_h/np.sum(f_h)
# #
# # out1=my_imfilter(im1, f)
# # out2=im2-my_imfilter(im2, f)+.5
#
# low_freq, high_freq, hybrid_im = create_hybrid_image(im1, im2, f,f_h)
#
#
# # import matplotlib.pyplot as plt
# # plt.imshow(f)
# # plt.show()
# # print(im)
# # k, l = input().split()
# # my_imfilter(im, ):
#
einstein = load_image("bird.bmp")
marilyn = load_image("plane.bmp")

hybrid = hybridImage(einstein, marilyn, 15, 10)
misc.imsave("dog-cat.png", np.real(hybrid))
