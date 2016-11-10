% Geometry signal
function Geo_sig = Geometric(N,n,a)
    % Verify
    if (n>N)
        disp('Error. Please input again');
    else
        % Define Step signal
        Geo_sig=zeros(1,N);
        for k=n:N
            Geo_sig(1,k)=a^(k-n);
        end
    end
%     % Plot the figure
%     figure;
%     stem(Geo_sig);
%     title('Geometric signal from n=10 and a=2');
%     xlabel('x');
%     ylabel('y');
end