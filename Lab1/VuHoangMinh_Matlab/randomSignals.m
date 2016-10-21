%---------------------------------------------------------------------
% Main function
%---------------------------------------------------------------------
function randomSignals()
    % Generate Gaussian Noise
    N=1000;
    gaussianX=generateGaussianNoise(N); 
    % Generate White Noise
    N=1000;
    whiteX=generateWhiteNoise(N);
    % Compute the autocorrelation
    computeAutoCorrelation(gaussianX,whiteX);
    % Compute the crosscorrelation
    computeCrossCorrelation();
end

%%
%---------------------------------------------------------------------
% Generate signals
%---------------------------------------------------------------------

% Generating Gaussian noise
function X_g=generateGaussianNoise(N) 
    % Generate X, and make computations
    X_g = randn(1,N);         
    meanX = mean(X_g);        
    stdX = std(X_g);          
    [countX, centerX] = hist(X_g);    
    stepX = centerX(2)-centerX(1);
    countX = countX/(sum(countX)*stepX);
    
    % Create Gaussian Distribution
    gaussianX = exp(-0.5*((centerX-meanX)/stdX).^2);
    gaussianX = gaussianX * (1 / (stdX * sqrt(2*pi)));
    
    % Plot the figure
    figure;
    plot(centerX, countX, centerX, gaussianX);
    legend('Distribution of data','Theoretical distribution');
    title(['Histogram of Data and Theoretical distribution at N=',num2str(N)]);
    xlabel('x');
    ylabel('y');
end

% Generating White noise
function X_u=generateWhiteNoise(N) 
    % Generate X, and make computations
    X_u = rand(1,N);
    meanX = mean(X_u);
    stdX = std(X_u);
    [countX, centerX] = hist(X_u);
    stepX = centerX(2)-centerX(1);
    countX = countX/(sum(countX)*stepX);
    
    % Create White Distribution
    uniformX(1:length(centerX)) = 1*max(countX); 
    
    % Plot the figure
    figure;
    plot(centerX, countX, centerX, uniformX);
    legend('Distribution of data','Theoretical distribution');
    title(['Histogram of Data and Theoretical distribution at N=',num2str(N)]);
    xlabel('x');
    ylabel('y');
end

% Compute the autocorrelation
function computeAutoCorrelation(gaussianX,whiteX)
    % Cross correlation of a gaussian noise 
    gaussianCorr = xcorr(gaussianX,'biased');
    uniformCorr = xcorr(whiteX, 'biased');
    
    % Plot the figure
    figure; 
    plot(gaussianCorr);
    title('White noise');
    figure; 
    plot(uniformCorr);
    title('Not White noise');
end

% Compute the crosscorrelation
function computeCrossCorrelation()
    % Generate 3 signals
    s1=round(rand(1,50));
    s2=round(rand(1,50));
    s3=round(rand(1,50)); 
    
    % Generate a whole signal Sig
    s(1:50)=s1;
    s(101:150)=s2;
    s(201:250)=s3;
    
    % Compute the cross-correlation
    s1Corr=xcorr(s,s1); 
    figure;
    plot(s1Corr);
    title('Correlation between S1 and S');
    s2Corr=xcorr(s,s2); 
    figure;
    plot(s2Corr);
    title('Correlation between S2 and S');
    s3Corr=xcorr(s,s3); 
    figure;
    plot(s3Corr);
    title('Correlation between S3 and S');
end