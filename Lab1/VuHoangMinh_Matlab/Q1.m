% Clean everything
clc;
clear all;
close all;

%-----------------------------Initiliazation-------------------------------
N=20;
n=10;
%Create 1 to N array
x = linspace(1,N,N);

%-----------------------------Question 1----------------------------------
% Make Dirac signal
Dirac_sig=Dirac(N,n);
% Plot the figure
figure;
stem(x,Dirac_sig);
title('Dirac signal at n=10');
xlabel('x');
ylabel('y');


%-----------------------------Question 2----------------------------------
% Make Step signal
Step_sig=Step(N,n);
% Plot the figure
figure;
stem(x,Step_sig);
title('Step signal from n=10');
xlabel('x');
ylabel('y');


%-----------------------------Question 3----------------------------------
a=2;
% Make Ramp signal
Ramp_sig=Ramp(N,n,a);
% Plot the figure
figure;
stem(x,Ramp_sig);
title('Ramp signal from n=10 and a=2');
xlabel('x');
ylabel('y');


%-----------------------------Question 4----------------------------------
a=2;
% Make Ramp signal
Geo_sig=Geometric(N,n,a);
% Plot the figure
figure;
stem(x,Geo_sig);
title('Ramp signal from n=10 and a=2');
xlabel('x');
ylabel('y');


%-----------------------------Question 5----------------------------------
a=3;
% Make Ramp signal
Box_sig=Box(N,n,a);
% Plot the figure
figure;
stem(x,Box_sig);
title('Box signal from n=10 and a=3');
xlabel('x');
ylabel('y');


%-----------------------------Question 6----------------------------------
% Make Sinfc signal
f=10;
Ts=100;
num_period=20;
x=1:num_period;
Sinfn_sig = Sinfn(f,Ts,num_period);
% Plot the figure
figure;
stem(x,Sinfn_sig);
title('Sinfc signal for f=10, fs=100');
xlabel('x');
ylabel('y');

function Dirac_sig=Dirac(N,n)
% Verify
if (n>N)
    disp('Error. Please input again');
else
    % Define Dirac signal
    Dirac_sig=zeros(N);
    Dirac_sig(n)=1;
end

