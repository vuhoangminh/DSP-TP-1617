%%
%=====================================================================
% Lab 2 Digital Signal Processing
%=====================================================================

function Lab2()

	%=====================================================================
	%Reminder 1: sin functions 
	%=====================================================================
	 fs = 20 ; %Sampling Frequency 
	 f = 1;  %Frequency 
	 t1 = [0:1/fs:10];  
	 y1 = sin(2*pi*f*t1); %Sin Function
	 figure(1)
	 subplot(2,1,1); plot(t1,y1); %Plotting Sin Function 
	 title('Sin Function')  
	 xlabel('t')
	 ylabel(' sin(2*pi*f*t1) ')

	 
	 t2 = [0:1/fs:20];
	 y2 = sin(2*pi*(f/fs)*t2);
	 subplot(2,1,2) ;  plot(t2,y2);  %Plotting Sin Function 
	 title('Sin Function')  
	 xlabel('t')
	 ylabel(' sin(2*pi*(f/fs)*t2)')
        
	%=====================================================================
	%Exercise 1 : Causality 
	%=====================================================================

	%---------------------------------------------------------------------
	%Part1.1 : Causality  

	x = step(4,20); % plotting Signal with Step of 4 to 20 
	for k= 1:19
		y(k) = x(k)/2+ (x(k+1))/2;
	end
	 
	%Non Causal System
	subplot(3,1,2); 
	stem(y);
	title('Non Causal System')  
	xlabel('x(k)')
	ylabel('y(k)')

	%---------------------------------------------------------------------
	%Part1.2 : Causality 

	%Making signal causal by shifting past by putting k-1 in 
	%the system which will take only past values
	for k= 2:20
		y(k) = x(k)/2+ (x(k-1))/2;
	end
	
	subplot(3,1,3); stem(y)% Causal System
	title('Causal System')  
	xlabel('x(k) ')
	ylabel(' y(k) ')
	

	%=====================================================================
	%Exercise 2 : Stability  
	%=====================================================================
	
	%---------------------------------------------------------------------
	%Part 2.1 : Stability
	y5 = x; %x is a step function
	for i = 2:1:20
		y5(i) = y5(i-1)+x(i); % Accumulating past results with current 
							  % results
	end
	figure(3)        
	subplot(2,1,1); stem(y5) % 
	title('Accumulation')  
	xlabel('x(k) ')
	ylabel('y(k)')
			
	%Part 2.2 :Stability
	 d = Dirac(4,20); %d is a step function
	 y6 = d;
	 for i = 2:1:20
		y6(i) = y6(i-1)+d(i);  
	end

	subplot(2,1,2); stem(y6)  % Plotting system with input as dirac 
	title('Stable System')  
	xlabel('d(k) ')
	ylabel('y(k)')
	  
	%---------------------------------------------------------------------
	%Part 2.3 :Stability
	 d = Dirac(4,20);
	 y7 = d;
	 for i = 2:1:20
		y7(i) = d(i)+ 2*(y7(i-1));
	 end

	figure(4)
	subplot(2,1,1); stem(y7)  % Plotting system with input as dirac 
	title('Unstable System')  
	xlabel('d(k) ')
	ylabel('y(k)')
		
	%---------------------------------------------------------------------
	%Part 2.4 :Stability
	 d = Dirac(4,20);
	 y8 = d;
	 for i = 2:1:20
		y8(i) = d(i)+ (y8(i-1)/3);
	end
	subplot(2,1,2); stem(y8)  % Plotting system with input as dirac 
	title('Stable System')  
	xlabel('d(n)')
	ylabel('y(k)')
	
	
%=====================================================================	
%Exercise 3 : Invariance and linearity  
%=====================================================================

	%---------------------------------------------------------------------
	%Part 3.1 : Invariance and linearity 
	
	xa=[0 0 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 0];
	xb=[0 0 0 0 0 0 0 0 0 4 3 2 1 0 0 0 0 0 0];
	ya(1)=0;
	yb(1)=0;


	for i=2:1:19-1
		 ya(i)=3*xa(i-1)-2*xa(i)+xa(i+1);
	end
	
	figure(5);
	subplot(2,1,1); stem(ya)  % Plotting system with input Xa 
	title('System Xa')  
	xlabel('x(a) ')
	ylabel('y(a)')

	%---------------------------------------------------------------------
	%%% Part 3.2  :Invariance and linearity 
	for i=2:1:19-1
		yb(i)=3*xb(i-1)-2*xb(i)+xb(i+1);
	end
	subplot(2,1,2); stem(yb)  % Plotting system with input Xb 
	title('System Xa')  
	xlabel('x(b) ')
	ylabel('y(b)')

	%---------------------------------------------------------------------
	%%% Part 3.3  : Invariance and linearity 
	h=[1,-2,3];
	x=xa+xb; %Adding two signals and then convolving
	y1=conv(x,h);
	figure(7)
	stem(y1);
	title('Convolution of two signal')  
	xlabel('X = conv(xa) + conv(xb) ')
	ylabel('y)')

	%---------------------------------------------------------------------
	%%% Part 3.4 :Invariance and linearity 
	y2=conv(xa,h)+conv(xb,h);  %Convolving first and then adding 
	figure(8)
	stem(y1);
	title('Convolution of two signal')  
	xlabel('X = conv(xa + xb) ')
	ylabel('y)')
end
%=====================================================================
%=====================================================================


%%
%=====================================================================
% Generate signals
%=====================================================================

%=====================================================================
% Dirac signal
function Dirac_sig =  Dirac(n,N) %Function Defination 

    if ((n<1)||(n>N))
            disp('Error : n should be inferior then N-1');  %Display error if n > N-1
            Dirac_sig= 0;
    else
            s = zeros(1,N);  
            s(n) = 1 ;
            Dirac_sig = s;
           
    end
  
end

%=====================================================================
% Step signal
function Step_sig =  step(n,N)  %Function Defination 

    if ((n<1)||(n>N))
            disp('Error : n should be inferior then N-1');  %Display error if n > N-1
            Step_sig= 0;
    else
            s = zeros(N,1);  
            for i = n+1:N
                s(i) = 1 ;
            end 
            Step_sig = s;
            
            figure(2)
            subplot(3,1,1)
            stem(Step_sig) ;  % Unit Step
            title('Step')  
            xlabel(' X')
            ylabel(' Y ')    
    end
end 