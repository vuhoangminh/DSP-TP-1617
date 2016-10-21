% Clean everything
clc;
clear all;
close all;

%-----------------------------Initiliazation-------------------------------
N=1000;

y = randn(1,N);
[h, xh] = hist(y);
dh = xh(2)-xh(1);
h = h/(sum(h)*dh);
% Compute mean and sigma
mu=mean(y);
sigma=std(y);
normal_dist=1/(sigma*sqrt(2*pi));
normal_dist=normal_dist*exp(-0.5*((xh-mu)./sigma).^2);

plot(xh,h);
hold off;
%plot(xh,normal_dist);
