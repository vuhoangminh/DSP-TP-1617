function Dirac_sig=Dirac(N,n)
% Verify
if (n>N)
    disp('Error. Please input again');
else
    % Define Dirac signal
    Dirac_sig=zeros(N);
    Dirac_sig(n)=1;
end