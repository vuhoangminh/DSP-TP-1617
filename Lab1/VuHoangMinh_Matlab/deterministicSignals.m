%---------------------------------------------------------------------
% Main function
%---------------------------------------------------------------------
function deterministicSignals()
%     % Initialize
%     N=20;
%     n=10;
%     a=2;
%     % Generate Dirac signal
%     Dirac(N,n);
%     % Generate Step signal
%     Step(N,n);
%     % Generate Ramp signal
%     Ramp(N,n,a);
%     % Generate Geometric signal
%     Geometric(N,n,a);    
%     % Generate Box signal
%     a=3;
%     Box(N,n,a);
    % Generate Sinfn signal
    F=10; Fs=30; N=2;
    Sinfc(F,Fs,N);
end

%%
%---------------------------------------------------------------------
% Generate signals
%---------------------------------------------------------------------

% Dirac signal
function Dirac(N,n)
    % Verify
    if (n>N)
        disp('Error. Please input again');
    else
        % Define Dirac signal
        Dirac_sig=zeros(N);
        Dirac_sig(n)=1;
    end
    % Plot the figure
    figure;
    stem(Dirac_sig);
    title('Dirac signal at n=10');
    xlabel('x');
    ylabel('y');
end

% Step signal
function Step(N,n)
    % Verify
    if (n>N)
        disp('Error. Please input again');
    else
        % Define Step signal
        Step_sig=zeros(N);
        for k=n:N
            Step_sig(k)=1;
        end
    end
    % Plot the figure
    figure;
    stem(Step_sig);
    title('Step signal from n=10');
    xlabel('x');
    ylabel('y');
end

% Ramp signal
function Ramp_sig = Ramp(N,n,a)
    % Verify
    if (n>N)
        disp('Error. Please input again');
    else
        % Define Step signal
        Ramp_sig=zeros(N);
        for k=n:N
            Ramp_sig(k)=(k-n)*a;
        end
    end
    % Plot the figure
    figure;
    stem(Ramp_sig);
    title('Ramp signal from n=10 and a=2');
    xlabel('x');
    ylabel('y');
end

% Geometry signal
function Geo_sig = Geometric(N,n,a)
    % Verify
    if (n>N)
        disp('Error. Please input again');
    else
        % Define Step signal
        Geo_sig=zeros(N);
        for k=n:N
            Geo_sig(k)=a^(k-n);
        end
    end
    % Plot the figure
    figure;
    stem(Geo_sig);
    title('Geometric signal from n=10 and a=2');
    xlabel('x');
    ylabel('y');
end

% Box signal
function Box(N,n,a)
    % Verify
    if (n<1+a || n>N-a)
        disp('Error. Please input again');
    else
        % Define Step signal
        Box_sig=zeros(N);
        for k=n-a:n+a
            Box_sig(k)=1;
        end
    end
    % Plot the figure
    figure;
    stem(Box_sig);
    title('Box signal from n=10 and a=3');
    xlabel('x');
    ylabel('y');
end

% Sincf signal
function Sinfc(F,Fs,N)
    % Compute period, sampling period, number of samples
    T=1/F;
    Ts=1/Fs;
    n=0:Ts:N*T;
    
    % Compute sin signal
    Sin_sig=sin(2*pi*F*n);
    
    % Plot the figure
    figure;
    stem(n,Sin_sig);
    title(['Sin of the signal F=',num2str(F),' Fs=',num2str(Fs),' N=',num2str(N)]);
    xlabel('x');
    ylabel('y');
end

