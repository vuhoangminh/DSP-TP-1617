% Dirac signal
function Dirac_sig=Dirac(N,n)
    % Verify
    if (n>N)
        disp('Dirac Error. Please input again');
    else
        % Define Dirac signal
        Dirac_sig=zeros(1,N);
        Dirac_sig(1,n)=1;
    end
%     % Plot the figure
%     figure;
%     stem(Dirac_sig);
%     title('Dirac signal');
%     xlabel('x');
%     ylabel('y');
end