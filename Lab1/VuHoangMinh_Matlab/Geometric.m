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
