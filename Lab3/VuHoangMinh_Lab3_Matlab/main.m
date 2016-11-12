%%==================================================
% Vu Hoang Minh, MAIA
% Lab 3 : Digital Signal Processing
%%==================================================

function main()
    % Initilization
    clc;close all;clear all;
	
	% Run Exercise Functions
%     exercise1();
%     exercise2();
    exercise3();
end

%=====================================================================
%Exercise 1
%=====================================================================
function exercise1()
% Initilization
x=[1 2 3 4];
N=5;
n=5;
a=exp(1);
numberPadding=4;
valuePadding=10;

%----------------------------------------------------------------------
% Generate filters
diracSignal = Dirac(N,n);
stepSignal = Step(N,n);
geometrySignal = Geometric(N,n,a);
randomSignal = [-1 1];

%----------------------------------------------------------------------
% Convolution filters with x
convDiracSignal = convolution(x,diracSignal);
convStepSignal = convolution(x,stepSignal);
convGeometrySignal = convolution(x,geometrySignal);
convRandomSignal = convolution(x,randomSignal);

%----------------------------------------------------------------------
% Generate extended signals
symmetrySignal = generatesymmetry(x,numberPadding);
periodicSignal = generateperiodic(x,numberPadding);
constantpaddingSignal = generateconstantpadding(x,numberPadding,valuePadding);
end

%=====================================================================
%Exercise 2
%=====================================================================
function exercise2()

%----------------------------------------------------------------------
% Gaussian kernel
K=[ 1 4 6 4 1;
    4 16 24 16 4;
    6 24 36 24 6;
    4 16 24 16 4;
    1 4 6 4 1
    ];
K=K/256;

%----------------------------------------------------------------------
% Read and show original image
inputImage=imread('lena-grey.bmp');
figure;
imshow(inputImage);
title('Lena original');

% Convolve Lena image with Gaussian kernel and display the output
convImage=convolution2D(inputImage,K);
figure;
imshow(convImage,[]);
title('Convolution of Lena original and Gaussian kernel');

%----------------------------------------------------------------------
% Define Sobel kernels
xSobelKernel=[-1 0 1; -2 0 2; -1 0 1];
ySobelKernel=[-1 -2 -1; 0 0 0; 1 2 1];

% Convolve Lena image with 2 Sobel kernels
xDerivativeImage=convolution2D(inputImage,xSobelKernel);
yDerivativeImage=convolution2D(inputImage,ySobelKernel);

% Compute Gradient direction
for i=1:size(inputImage,1)
    for j=1:size(inputImage,2)
        Theta(i,j)= yDerivativeImage(i,j)/xDerivativeImage(i,j);
    end
end

% Compute Gradient magnitude
for i=1:size(inputImage,1)
    for j=1:size(inputImage,2)
        G(i,j)=sqrt(xDerivativeImage(i,j)^2+yDerivativeImage(i,j)^2);
    end
end

% Display x, y and Gradient magnitude
figure;
imshow(xDerivativeImage,[]);
title('Horizontal derivative approximation');
figure;
imshow(yDerivativeImage,[]);
title('Vertical derivative approximation');
figure;
imshow(G,[]);
title('Gradient magnitude');
end

%=====================================================================
%Exercise 3
%=====================================================================
function exercise3()

%----------------------------------------------------------------------
% Read images: letter a and text
aImage = imread('a.png','png');
[textImage, cmap] = imread('text.png','png');

% Find thresholds and binarize images
thresholdAImage = graythresh(aImage);
binaryAImage = im2bw(aImage,thresholdAImage);
thresholdTextImage = graythresh(textImage);
binaryTextImage = im2bw(textImage,thresholdTextImage);

%----------------------------------------------------------------------
% Note: In fact, we can not use xcorr2 for the given TEXT_IMAGE without 
%       pre-processing, because the coordinates of 10 maximum values are 
%       not the coordinates of 'a'
%       So, we have to complement 'a' image to find the first 10 'a'

% Find cross-correlation of images
image1 = im2double(binaryAImage);
image1 = imcomplement(image1);
image2 = im2double(binaryTextImage);
crossCorrelation = xcorr2(image2, image1);
% Find the first 10 a
findtena(crossCorrelation, textImage, cmap);

%----------------------------------------------------------------------
% This is my solution before I realized that I have to complement 1 of
% these images
%----------------------------------------------------------------------
% Suggestion: I created another TEXT_IMAGE, namely text_2, and use 
%       normxcorr2 function for the original images instead of xcorr2 for
%       binary images. And it works!
% Find cross-correlation of images
[textImage, cmap] = imread('text_2.png','png');
crossCorrelation = normxcorr2(aImage(:,:,1), textImage(:,:, 1));
% Find the first 10 a
findtena(crossCorrelation, textImage, cmap);
end
