##==================================================
# Vu Hoang Minh, MAIA
# Lab 4 : Digital Signal Processing
##==================================================

# To make some nice plot
import matplotlib.pyplot as plt
# To play with arrays
import numpy as np

from skimage import io
from skimage import img_as_float
from matplotlib import cm
from scipy.stats import norm
from scipy import signal
import PIL
from PIL import Image
from scipy.misc import imresize
import cv2
from skimage import img_as_ubyte
from skimage.color import rgb2gray ,gray2rgb
from skimage.io import imread, imshow
from skimage.transform import rescale
import os, os.path



# ===============================================================
# Exercise 1 : Discrete Fourier Transform =======================
# ===============================================================

# ---------------------------------------------------------------
# Initialization
f = 5.
fs = 50.
t = np.arange(0, 1., 1./fs)     # Time vector for one second
N = 1000                        # Number of samples

# ---------------------------------------------------------------
# Sin Signal
x_n = np.sin(2*np.pi*f*t)
fr = (np.arange(-N/2,N/2,1)) * fs/N # frequency vector
# DFT
# Using fftshift to have the center frequency
x_f = np.fft.fftshift(np.fft.fft(x_n, N))

# Plot Signal
plt.figure()
plt.subplot(311)
plt.plot(t, x_n)
plt.title('Sin wave Signal')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude')

plt.subplot(312)
plt.plot(fr, x_f)
plt.title('DFT')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')

plt.subplot(313)
plt.plot(fr, np.abs(x_f))
plt.title('Magnitude')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')

plt.figure()
plt.subplot(211)
plt.plot(fr, np.real(x_f))
plt.title('Real')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Re(X(f))')

plt.subplot(212)
plt.plot(fr, np.imag(x_f))
plt.title('Imaginary')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Im(X(f))')


# ---------------------------------------------------------------
# Cos Signal
x_n = np.cos(2*np.pi*f*t)
# frequency vector
fr = (np.arange(-N/2,N/2,1)) * fs/N
# DFT
# Using fftshift to have the center frequency
x_f = np.fft.fftshift(np.fft.fft(x_n, N))

# Plot Signal
plt.figure()
plt.subplot(311)
plt.plot(t, x_n)
plt.title('Cos wave Signal')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude')

plt.subplot(312)
plt.plot(fr, x_f)
plt.title('DFT')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')

plt.subplot(313)
plt.plot(fr, np.abs(x_f))
plt.title('Magnitude')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')

plt.figure()
plt.subplot(211)
plt.plot(fr, np.real(x_f))
plt.title('Real')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Re(X(f))')

plt.subplot(212)
plt.plot(fr, np.imag(x_f))
plt.title('Imaginary')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Im(X(f))')

# Discussion: Cos and Sin differ only by phase shift



# ---------------------------------------------------------------
# Square wave Signal
x_n = signal.square(2*np.pi*f*t)
# frequency vector
fr = (np.arange(-N/2,N/2,1)) * fs/N
# DFT
# Using fftshift to have the center frequency
x_f = np.fft.fftshift(np.fft.fft(x_n, N))


plt.figure()
plt.subplot(311)
plt.plot(t, x_n)
plt.title('Square wave Signal')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude')

plt.subplot(312)
plt.plot(fr, x_f)
plt.title('DFT')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')

plt.subplot(313)
plt.plot(fr, np.abs(x_f))
plt.title('Magnitude')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')

plt.figure()
plt.subplot(211)
plt.plot(fr, np.real(x_f))
plt.title('Real')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Re(X(f))')

plt.subplot(212)
plt.plot(fr, np.imag(x_f))
plt.title('Imaginary')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Im(X(f))')



# ---------------------------------------------------------------
# Define the mean and standard deviation
mu = 0.
sigma = 1.
N = 10000
t = np.arange(0, 1., 1./N)
# Generate the data
x_n = np.random.normal(mu, sigma, (N, ))

# In fact, the Fourier transform of white noise is... white noise!

plt.figure()
plt.plot(t, x_n)
plt.title('Random Signal')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude')




# ===============================================================
# Exercise 2 -  Sampling ========================================
# ===============================================================

# ---------------------------------------------------------------
# Define some functions
def getsignal (f):      # generate signal
    f1 = 5
    f2 = 20
    t = np.arange(0, 1., 1. / f)
    x_n = 3*np.cos(2*np.pi*f1*t) + 4*np.cos(2*np.pi*f2*t)
    return x_n, t

def plotxn (x_n, t):    # plot signal xn
    plt.figure()
    plt.plot(t, x_n)
    plt.title('Signal for fs=%d' %f)
    plt.xlabel('Time(sec)')
    plt.ylabel('Amplitude')

def computefft(fs,x_n):
    # Number of samples
    N = 1000
    # frequency vector
    fr = (np.arange(-N/2,N/2,1)) * fs/N
    # DFT
    # Using fftshift to have the center frequency
    x_f = np.fft.fftshift(np.fft.fft(x_n, N))
    return fr, x_f

def plotfft (fr, x_f):
    plt.figure()
    plt.plot(fr, x_f)
    plt.title('DFT for fs=%d' % f)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('|X(f)|')


# ---------------------------------------------------------------
# Plot $x[n]$ for different sampling frequencies in time domain.
fs = [10, 20, 25, 40, 50, 100, 150]
for f in fs:
    x_n, t = getsignal(f)
    plotxn (x_n, t)


# ---------------------------------------------------------------
# Compute the FFT of the above signals and display their centered frequency components
for f in fs:
    x_n, t = getsignal(f)
    fr, x_f = computefft(f,x_n)
    plotfft (fr, x_f)


# ---------------------------------------------------------------
# Discuss your observations.
#       Aliasing occurs when fs < 2 max(f1,f2)
#       Aliasing is an effect that causes different signals to become indistinguishable
#           (or aliases of one another) when sampled
#       Clearly, with not sufficient fs, we can not distinguish signal 1 and 2




# ===============================================================
# Exercise 3 - 1D  DFT for image classification =================
# ===============================================================

# ---------------------------------------------------------------
# Define some functions
# Normalize image to obtain max intensity is 255
def normimage (image):
    image_ubyte = img_as_ubyte(image)
    image_gray = rgb2gray(image_ubyte)
    maxIntensity = image_gray.max()
    image_normalized = 255.0 / maxIntensity * np.array(image_gray)
    return image_normalized

# Resize image to the same width
def resizeimage (image, minColumn):
    hsize = minColumn
    img = image
    nRow, nColumn = image_normalized.shape
    wpercent = float(nColumn)/float(nRow)
    basewidth = int(float(hsize/wpercent))
    # image_resized = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    image_resized = imresize(img, (basewidth, hsize), interp='bicubic')
    return image_resized

# Find the smallest width of all images to resize later
def findminwidth (pathImage):
    nFilename = sorted([iFilename for iFilename in os.listdir(pathImage)])
    minColumn = 0
    for iFilename in nFilename:
        image = io.imread(os.path.join(Path, iFilename))
        if len(image.shape)>2:
            row, column, colorabc = image.shape
        else:
            row, column = image.shape
        if minColumn == 0 :
            minColumn = column
        else:
            if minColumn > column:
                minColumn = column
    return minColumn

# Compute fft of a specific row of an image
def computefftrow (image, numRow):
    # Initilization
    image_ubyte = img_as_ubyte(image)
    image_gray = rgb2gray(image_ubyte)
    image_gray_1D = image_gray[numRow,:]
    # Number of samples
    N = 1000
    # Using fftshift to have the center frequency
    x_f = np.fft.fftshift(np.fft.fft(image_gray_1D, N))
    return x_f

# Plot fft of x_f
def plotfft (N, x_f):
    fs = 50
    fr = (np.arange(-N / 2, N / 2, 1)) * fs / N
    plt.figure()
    plt.plot(fr, x_f)
    plt.title('DFT')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('|X(f)|')

# Plot fft of x_f
def computeaccuracy (numTrue,numImage):
    return float(100*numTrue/numImage)


# ---------------------------------------------------------------
# Read 'profile1.png'
Path = './'
inputImage = 'profile1.png'
profileImage = imread(Path.__add__(inputImage))
profileImage_ubyte = img_as_ubyte(profileImage)
profileImage_gray=rgb2gray(profileImage_ubyte)

# Plot read image
plt.figure()
imshow(profileImage_gray)
plt.title('Gray profile')
plt.axis('off')

# Plot read image
profileImage_gray_1D = profileImage_gray.flatten()
profileImage_fft = np.fft.fft(profileImage_gray_1D)

# Number of samples
N = 1000
fs = 50
# frequency vector
fr = (np.arange(-N/2,N/2,1)) * fs/N
# DFT
# Using fftshift to have the center frequency
x_f = np.fft.fftshift(np.fft.fft(profileImage_gray_1D, N))

# Plot
plt.figure()
plt.plot(fr, x_f)
plt.title('DFT of Profile1')
plt.xlabel('Frequency(Hz)')
plt.ylabel('|X(f)|')


# ---------------------------------------------------------------
# Seprate the images to two different groups of \barcode" and \non-barcode" based on their frequency
# spectrum of their profile

# Initialization
Path = '../images/1D-DFT/'
nFilename = sorted([iFilename for iFilename in os.listdir(Path)])
numFile = len(nFilename)

# Bar-code image array: I am using this array to compute my accuracy rate
matrixTrue = np.array([1, 2, 6, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54])
# Number of true images detected i.e barcode to barcode and non-barcode to non-barcode
numTrue = 0


# ---------------------------------------------------------------
# Observation: I am using an offset to differentiate whether an image is barcode or not
# Algorithm: I realized 2 factors would affect the offset:
#            (1) The magnitude of centered DFT of 1D image row
#            (2) The difference between the DFT of 2 rows, let say, 5 rows away the middle row
#                   In specific, in barcode image, the DFT of neighboring row is nearly similar
#                   while in non-barcode image, there is a big difference

# Factor 1, factor 2 arrays
fac1Array = np.zeros(54)
fac2Array = np.zeros(54)
# File number for debugging
iFile = 0

# Offset to differentiate bar and non-bar code
offset=1.3*2500000

# Find minwidth
# Note: Actually, I prefer to set at 100
Path = '../images/1D-DFT/'
minColumn=findminwidth (Path)
minColumn=100

# For each image
for iFilename in nFilename:
    # Read image
    iFile = iFile+1
    image = io.imread(os.path.join(Path, iFilename))
    if len(image.shape)>2:
        row, column, colorabc = image.shape
    else:
        row, column = image.shape
    print ("Size of original image: %d %d" %(row, column))
    # Normalize image
    image_normalized = normimage (image)
    row, column = image_normalized.shape
    print ("Size of original image: %d %d" % (row, column))
    # Resize image
    image_resized = resizeimage(image_normalized, minColumn)
    row, column = image_resized.shape
    print ("Size of original image: %d %d" % (row, column))
    print iFile

    # Compute DFT of 3 rows: mid, mid-5 and mid+5 - refer to Observation (2)
    x_f = computefftrow(image_resized,int(row/2))
    x_f1 = computefftrow(image_resized,int(row/ 2 - 5))
    x_f2 = computefftrow(image_resized,int(row/ 2 + 5))

    # Compute factor 1 and store
    fac1Score = np.abs(max(x_f))
    print fac1Score
    fac1Array[iFile - 1] = fac1Score

    # Compute factor 2 and store
    fac2Score = abs(np.abs(max(x_f)) - np.abs(max(x_f1))) + abs(np.abs(max(x_f)) - np.abs(max(x_f2)))
    print fac2Score
    fac2Array[iFile - 1] = fac2Score

    # Compute final score of image
    finalScore = fac1Score * fac2Score
    # Offset here to help separate
    finalScore = finalScore / offset
    print finalScore

    # If finalScore < 1, this image is Barcode
    if finalScore < 1:
        print ('YES')
        # Increase numTrue by 1 if correct
        if iFile in matrixTrue:
            numTrue = numTrue + 1
        else:
            numTrue = numTrue
    else:
        print ('NO')
        # Increase numTrue by 1 if correct
        if iFile in matrixTrue:
            numTrue = numTrue
        else:
            numTrue = numTrue + 1
    print ('==============================================')

print ('The number of true detected images is: ')
print numTrue
accurateRate=computeaccuracy (numTrue,54)
print ('The accuracy rate is (%): ')
print accurateRate


# ---------------------------------------------------------------
# Note: In fact, we can improve the result by pre and pst processing
#		However, it will take time, so I stop here 


# ===============================================================
# Show plots ====================================================
# ===============================================================
plt.show()