function Box_sig = Box(N,n,a)
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
