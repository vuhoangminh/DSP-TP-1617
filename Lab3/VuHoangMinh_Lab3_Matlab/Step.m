% Step signal
function Step_sig=Step(N,n)
    % Verify
    if (n>N)
        disp('Error. Please input again');
    else
        % Define Step signal
        Step_sig=zeros(1,N);
        for k=n:N
            Step_sig(1,k)=1;
        end
    end
%     % Plot the figure
%     figure;
%     stem(Step_sig);
%     title('Step signal from n=10');
%     xlabel('x');
%     ylabel('y');
end