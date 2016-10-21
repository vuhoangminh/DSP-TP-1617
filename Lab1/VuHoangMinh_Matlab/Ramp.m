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
