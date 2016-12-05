import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('lena.png')
print("color image shape", img.shape) # 512 x 512 x 3
plt.imshow(img)
plt.show()

# make it B&W
bw = img.mean(axis=2) # we are taking the mean of the RGB values?
print("bw image shape", bw.shape) # 512 x 512
plt.imshow(bw, cmap='gray')
plt.show()

# create a Gaussian filter
W = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2 # filter indices are 0..19,0..19; 9.5 is 19 / 2
        W[i, j] = np.exp(-dist / 50.)

# let's see what the filter looks like
print("filter shape", W.shape)
plt.imshow(W, cmap='gray')
plt.show()

# now the convolution
out = convolve2d(bw, W)
plt.imshow(out, cmap='gray')
plt.show()

print("orignal image shape", bw.shape) # 512 x 512
print('shape after convolution', out.shape) # 531 x 531
# after convolution, the output signal is N1 + N2 - 1, where N1xN1 is image size, N2xN2 is filter size

# we can also just make the output the same size as the input
out = convolve2d(bw, W, mode='same')
plt.imshow(out, cmap='gray')
plt.show()
print("shape after mode='same' convolution", out.shape)

# in color
out3 = np.zeros(img.shape)
for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
plt.imshow(out3)
plt.show()
