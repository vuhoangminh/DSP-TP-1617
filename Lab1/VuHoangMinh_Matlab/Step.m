function Step_sig = Step(N,n)
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
