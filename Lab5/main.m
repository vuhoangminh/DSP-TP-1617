%%==================================================
% Vu Hoang Minh, MAIA
% Lab 5 : Digital Signal Processing
%%==================================================


function main()
% Initilization
clc;close all;clear all;

% Run Exercise Functions
exercise1();
exercise2();
exercise3();
exercise4();
end

%=====================================================================
%Exercise 1
%=====================================================================
function exercise1()

% ---------------------------------------------------------------
% Initilize synthetic image
img = zeros(301,301);
img(100:200, 140:160) = 255;
figure;
imshow(img);

% Compute the FFT
imgFreq_fft = fftshift(fft2(img));
figure();
subplot(121); imagesc(abs(imgFreq_fft)); colormap('gray'); title('Magnitude');
subplot(122); imagesc(angle(imgFreq_fft)/pi*180); colormap('gray'); title('Phase');

% ---------------------------------------------------------------
% Compute translated, rotated images
img = zeros(301,301);
img(100:200, 140:160) = 255;
imgTrans = zeros(301,301);
imgTrans(150:250, 160:180) = 255;
imgRot = imrotate(img, 45);
img2 = zeros(301,301);
img2(20:120, 140:160) = 255;
img2(180:280, 140:160) = 255;
img3 = zeros(301,301);
img3(100:200, 145:155) = 255;

% ---------------------------------------------------------------
% Compute the FFT of translated image and draw
imgTrans_fft = fftshift(fft2(imgTrans));
figure;
imshow(imgTrans,[]);
title('Translated image');
figure;
subplot(121); imagesc(abs(imgTrans_fft)); colormap('gray'); title('Magnitude');
subplot(122); imagesc(angle(imgTrans_fft)/pi*180); colormap('gray'); title('Phase');

% ---------------------------------------------------------------
% Compute the FFT of rotated image and draw
imgRot_fft = fftshift(fft2(imgRot));
figure;
imshow(imgRot,[]);
title('Rotated image');
figure;
subplot(121); imagesc(abs(imgRot_fft)); colormap('gray'); title('Magnitude');
subplot(122); imagesc(angle(imgRot_fft)/pi*180); colormap('gray'); title('Phase');

% ---------------------------------------------------------------
% Compute the FFT of image2 and draw
imgFreq2_fft = fftshift(fft2(img2));
figure;
imshow(img2,[]);
title('Image 2');
figure;
subplot(121); imagesc(abs(imgFreq2_fft)); colormap('gray'); title('Magnitude');
subplot(122); imagesc(angle(imgFreq2_fft)/pi*180); colormap('gray'); title('Phase');

% ---------------------------------------------------------------
% Compute the FFT of image3 and draw
imgFreq3_fft = fftshift(fft2(img3));
figure;
imshow(img3,[]);
title('Image 3');
figure();
subplot(121); imagesc(abs(imgFreq3_fft)); colormap('gray'); title('Magnitude');
subplot(122); imagesc(angle(imgFreq3_fft)/pi*180); colormap('gray'); title('Phase');


% ---------------------------------------------------------------
% Observation:
%   Translation: have effect on the the phase of FFT but not the magnitude 
%   Rotation: have effect on the phase and magnitude of FFT at the same
%       time
%   Image 2: If we add another object, the magnitude of FFT will increase
%		In addition, there is an effect on the phase, but hard to realize
%   Image 3: If the width of the object is decreased, the orientation of 
%		|F(u;0)| and |F(0;v)| is shifted by 90 degree. And the phase is affeted too

end



%=====================================================================
%Exercise 2
%=====================================================================
function exercise2()

%----------------------------------------------------------------------
% Consider another synthetic image
Im=0;
N=64;
T=1;
Ts=T/N;
Fs=1/Ts;
df=Fs/N;
Im(N/8:N/4,N/4+1:N/2)=1;
Im(1:N/4,N/2+1:N)=Im;
Im(N/4+1:N/2,:) = Im;
Im(N/2+1:3*N/4,:) = Im(1:N/4,:);
Im(3*N/4+1:N,:) = Im(1:N/4,:);

% Compute FFT, F(u,0) and F(0,v)
imgFreq_fft = fftshift(fft2(Im));
fu0 = imgFreq_fft(N/2+1,:);
f0v = imgFreq_fft(:,N/2+1);
fr = (-N/2 : N/2-1);

% Plot FFT
figure;
imshow(Im,[]);
title('Another synthetic image');
figure();
subplot(121); imagesc(abs(imgFreq_fft)); colormap('gray'); title('Magnitude');
subplot(122); imagesc(angle(imgFreq_fft)/pi*180); colormap('gray'); title('Phase');

% Plot FFT of fu0
figure; 
subplot(121); 
plot(fr,abs(fu0));  
title('Magnitude of f(u,0)')
subplot(122); 
plot(fr,angle(fu0)/pi*180);  
title('Phase of f(u,0)');

% Plot FFT of fv0
figure; 
subplot(121);
plot(fr,abs(f0v));  
title('Magnitude of f(0,v)');
subplot(122); 
plot(fr,angle(f0v)/pi*180);  
title('Phase of f(0,v)');

end



%=====================================================================
%Exercise 3
%=====================================================================
function exercise3()
%----------------------------------------------------------------------
% Read and show original image
inputImage=imread('./images/lena-grey.bmp');
figure;
imshow(inputImage);
title('Lena original');

% Compute FFT of image
imgFreq_fft=fft2(inputImage);

% Extract magnitude and phase
mag1=abs(imgFreq_fft);
s=log(1+fftshift(imgFreq_fft));
phase1=angle(imgFreq_fft);

% Reconstruct image by phase or magnitude
r1=ifftshift(ifft2(mag1));
r2=ifft2(exp(1i*phase1));

% Plot
figure; 
subplot(121);
imshow(mag1,[]);
colormap('gray'); title('Magnitude'); 
title('Magnitude');
subplot(122); 
imshow(phase1,[]);
title('Phase');

% Reconstruct by magnitude
% Observation: I get nothing
figure;
imshow(uint8(r1));
title('Reconstructed by magnitude');

% Reconstruct by phase
% Observation: the overall information can be reconstructed
figure;
imshow(r2,[]);
title('Reconstructed by phase');

end

% The magnitude of the FFT is like how much energy there is in the sine 
% waves used to build up your image. The phase is like how those sine 
% waves are positioned. For a real, unsymmetrical image, your FFT will be Hermitian. 



%=====================================================================
%Exercise 4
%=====================================================================
function exercise4()

% ---------------------------------------------------------------
% Sobel vertical matrix
Gx=[-1 0 1;
    -2 0 2;
    -1 0 1
    ];

% Read and plot image
inputImage=imread('./images/lena-grey.bmp');
figure;
imshow(inputImage);
title('Lena original');

% Convolve with Gx and plot
Ix = conv2(inputImage,Gx);
figure; 
imshow(Ix,[]);
title('Convolution in time domain');


% ---------------------------------------------------------------
% Find paddesize of image
sizeImage = size(inputImage);
sizeKernel = size(Gx);
sizePaddedImage = sizeImage + sizeKernel - 1;

% Find FFT of original image and kernel
F = fft2(double(inputImage), sizePaddedImage(1), sizePaddedImage(2));
H = fft2(double(Gx), sizePaddedImage(1), sizePaddedImage(2));

% Multiplication in Frequency domain
F_fH = H.*F;
ffi = ifft2(F_fH);

% Crop the image at its original size
ffi = ffi(2:size(inputImage,1)+1, 2:size(inputImage,2)+1);

%Display results (show all values)
figure;
imshow(ffi,[]);
title('Multiplication in frequency domain');

end