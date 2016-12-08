##==================================================
# Vu Hoang Minh, MAIA
# Lab 6&7 : Digital Signal Processing
##==================================================

import matplotlib.pyplot as plt
# To play with arrays
import numpy as np
import scipy as sp
from skimage import io

from skimage import img_as_float
from matplotlib import cm
from skimage import img_as_ubyte
from scipy import signal



# ===============================================================
# Functions =====================================================
# ===============================================================

# ---------------------------------------------------------------
# generate dirac
def dirac(n=0, N=10):
    if n < 0 or n > N - 1:
        raise NameError('n should be in range [0, N-1]')
        return

    dirac_seq = np.zeros((N,))
    dirac_seq[n] = 1

    return dirac_seq

# ---------------------------------------------------------------	
# generate step
def step(n=0, N=20):
    if n < 0 or n > N - 1:
        raise NameError('n should be in range [0, N-1]')
        return

    step_seq = np.zeros((N,))
    step_seq[n:] = np.ones((N - n,))

    return step_seq

# ---------------------------------------------------------------	
# smoothing recursive filtering
def smoothingrecursive(x, scaling, Ts):
    #Normalize scaling to Ts
    scaling = scaling/Ts

    #first we pad x with 1 zero on the front and two on the back
    x = np.pad(x, [2, 2], 'constant')

    #Smoothing recursive filter
    alpha = scaling*Ts
    a = np.exp(-alpha)

    y_causal = np.zeros(x.shape)
    for k in range(2,x.size):
        y_causal[k] = x[k] + a*(alpha - 1)*x[k-1] + 2*a*y_causal[k-1] - a*a*y_causal[k-2]

    y_anticausal = np.zeros(x.shape)
    for k in range(x.size-3, -1, -1):
        y_anticausal[k] = a*(alpha + 1)*x[k+1] - a*a*x[k+2] + 2*a*y_anticausal[k+1] - a*a*y_anticausal[k+2]

    y = y_causal + y_anticausal

    #Remove padding and return
    return y[2:-2], y_causal[2:-2], y_anticausal[2:-2]

# ---------------------------------------------------------------	
# plot smoothing recursive result
def plotsmoothingrecursive(x, scaling, Ts):
    y, y_causal, y_anticausal = smoothingrecursive(x, scaling, Ts)
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.title('Original signal')
    plt.stem(x)

    plt.subplot(2, 2, 2)
    plt.title('Recursive smoothing with scaling %.1f' % scaling)
    plt.stem(y)

    plt.subplot(2, 2, 3)
    plt.title('Causal part of smoothing')
    plt.stem(y_causal)

    plt.subplot(2, 2, 4)
    plt.title('Anti-causal part of smoothing')
    plt.stem(y_anticausal)

# ---------------------------------------------------------------	
# derivative recursive filtering
def derivativerecursive(x, scaling, Ts):
    #Normalize scaling to Ts
    scaling = scaling/Ts

    #first we pad x with 1 zero on the front and two on the back
    x = np.pad(x, [2, 2], 'constant')

    #Derivative recursive filter
    alpha = scaling*Ts
    a = np.exp(-alpha)

    y_causal = np.zeros(x.shape)
    for k in range(2,x.size):
        y_causal[k] = -scaling*a*alpha*x[k-1] + 2*a*y_causal[k-1] - a*a*y_causal[k-2]

    y_anticausal = np.zeros(x.shape)
    for k in range(x.size-3, -1, -1):
        y_anticausal[k] = scaling*a*alpha*x[k+1] + 2*a*y_anticausal[k+1] - a*a*y_anticausal[k+2]

    y = y_causal + y_anticausal

    #Remove padding and return
    return y[2:-2], y_causal[2:-2], y_anticausal[2:-2]

# ---------------------------------------------------------------	
# plot derivative recursive result
def plotderivativerecursive(x, scaling, Ts):
    y, y_causal, y_anticausal = derivativerecursive(x, scaling, Ts)
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.title('Original signal')
    plt.stem(x)

    plt.subplot(2, 2, 2)
    plt.title('Derivative smoothing with scaling %.1f' % scaling)
    plt.stem(y)

    plt.subplot(2, 2, 3)
    plt.title('Causal part of smoothing')
    plt.stem(y_causal)

    plt.subplot(2, 2, 4)
    plt.title('Anti-causal part of smoothing')
    plt.stem(y_anticausal)


# ===============================================================
# Exercise 1 : Filtering ========================================
# ===============================================================

# ---------------------------------------------------------------
# 1.1 Butterworth
filter_order = 3
w_cutoff = 0.4
w_pass = 0.7

# Compute
# lowpass
B1, A1 = signal.butter(filter_order, w_cutoff, btype='lowpass')
W1, H1 = signal.freqz(B1, A1)

# highpass
B2, A2 = signal.butter(filter_order, w_pass, btype='highpass')
W2, H2 = signal.freqz(B2, A2)

# bandpass
B3, A3 = signal.butter(filter_order, [w_cutoff, w_pass], btype='bandpass')
W3, H3 = signal.freqz(B3, A3)

# bandstop
B4, A4 = signal.butter(filter_order, [w_cutoff, w_pass], btype='bandstop')
W4, H4 = signal.freqz(B4, A4)

# Plot
plt.figure()

# lowpass
plt.subplot(2,2,1)
plt.title('Butterworth LPF at cutoff %.1f order %d' %(w_cutoff,filter_order))
plt.plot(W1/np.pi, np.abs(H1))
plt.ylabel('Gain'), plt.xlabel('Normalized Frequency')
plt.ylim([-0.1,1.1])

# highpass
plt.subplot(2,2,2)
plt.title('Butterworth HPF at cutoff %.1f order %d' %(w_cutoff,filter_order))
plt.plot(W2/np.pi, np.abs(H2))
plt.ylabel('Gain'), plt.xlabel('Normalized Frequency')
plt.ylim([-0.1,1.1])

# bandpass
plt.subplot(2,2,3)
plt.title('Butterworth BPF from %.1f to %.1f order %d' %(w_cutoff,w_pass,filter_order))
plt.plot(W3/np.pi, np.abs(H3))
plt.ylabel('Gain'), plt.xlabel('Normalized Frequency')
plt.ylim([-0.1,1.1])

# bandstop
plt.subplot(2,2,4)
plt.title('Butterworth BSF from %.1f to %.1f order %d' %(w_cutoff,w_pass,filter_order))
plt.plot(W4/np.pi, np.abs(H4))
plt.ylabel('Gain'), plt.xlabel('Normalized Frequency')
plt.ylim([-0.1,1.1])

# plt.show()

# ---------------------------------------------------------------
# 1.2 Low-pass Butterworth at different order
filter_order = [3, 5, 10, 20]
c = ['r','b','g','k']
w_cutoff = 0.4
w_pass = 0.7
max_ripple_db = 1.0

# Compute

plt.figure()
for i in range (0,len(filter_order)):
    B, A = signal.cheby1(filter_order[i], max_ripple_db, w_cutoff, btype='lowpass')
    W, H = signal.freqz(B, A)
    plt.plot(W/np.pi , np.abs(H) , c[i])

plt.legend(['order 3' , 'order 5' , 'order 10' , 'order 20'])
plt.title('Butterworth LPF with increasing order')


# Note: tradeoff between transition and attenuation
# Try order = 40 -> impossible, because > 1 it is wrong


# ===============================================================
# Exercise 2 : Recursive Filtering ==============================
# ===============================================================

# ---------------------------------------------------------------
# 2.1 Dirac
x_dirac = dirac(20, 41)
plt.figure()
plt.stem(x_dirac)
plt.title('Dirac signal')

# ---------------------------------------------------------------
# 2.2 Smoothing filter on Dirac function
# scaling 2
scaling = 2
Ts = 1/41
x = x_dirac
plotsmoothingrecursive(x, scaling, Ts)

# scaling 0.5
scaling = 0.5
plotsmoothingrecursive(x, scaling, Ts)


# ---------------------------------------------------------------
# 2.3 Box function
x_box = step(10, 41) - step(30, 41)
plt.figure()
plt.title('Box signal')
plt.stem(x_box)

# ---------------------------------------------------------------
# 2.4 Derivative filter on Dirac function
# scaling 2 for both filters
scaling = 2
Ts = 1/41
x = x_box
plotsmoothingrecursive(x, scaling, Ts)
plotderivativerecursive(x, scaling, Ts)

# scaling 0.5 for both filters
scaling = 0.5
plotsmoothingrecursive(x, scaling, Ts)
plotderivativerecursive(x, scaling, Ts)




# ===============================================================
# Exercise 3 : Canny-Deriche Filtering ==========================
# ===============================================================

# ---------------------------------------------------------------
# 3.1 Load and display image
image = img_as_float(io.imread('../images/boat.256.gif', as_grey=True))
plt.figure()
io.imshow(image)
plt.axis('off')
plt.title('Original image')

# ---------------------------------------------------------------
# 3.2 Apply the smoothing (derivative) Filter along the 
# columns (rows) of the images to obtain the component of the 
# gradient on the horizontal direction.
# 3.3 Apply the smoothing (derivative) Filter along the 
# rows (columns) of the images to obtain the component of the 
# gradient on the vertical direction.

# initilization
scaling_smoothing = 1.5
scaling_derivative = 1.5
Ts = 1/41

# apply smoothing filter in rows
image_hor_smoothed = np.zeros(image.shape) 
for row in range(image.shape[0]):
    y, y_causal, y_anticausal = smoothingrecursive(image[row,:], scaling_smoothing, Ts)
    image_hor_smoothed[row, :] = y
    
# apply derivative filter to get horizontal gradient
image_hor_grad = np.zeros(image.shape) 
for row in range(image.shape[0]):
    y, y_causal, y_anticausal = derivativerecursive(image_hor_smoothed[row,:], scaling_derivative, Ts)
    image_hor_grad[row, :] = y
    
# apply smoothing filter in cols
image_vert_smoothed = np.zeros(image.shape) 
for col in range(image.shape[1]):
    y, y_causal, y_anticausal = smoothingrecursive(image[:, col], scaling_smoothing, Ts)
    image_vert_smoothed[:, col] = y
    
# apply derivative filter to get vertical gradient
image_vert_grad = np.zeros(image.shape) 
for col in range(image.shape[1]):
    y, y_causal, y_anticausal = derivativerecursive(image_vert_smoothed[:,col], scaling_derivative, Ts)
    image_vert_grad[:, col] = y

# plot
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image_hor_grad, cmap='gray')
plt.title('Horizontal Gradient')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image_vert_grad, cmap='gray')
plt.title('Vertical Gradient')
plt.axis('off')

# ===============================================================
# Show plots ====================================================
# ===============================================================
plt.show()