function y3 =  step(n,N)  %Function Defination 

    if ((n<1)||(n>N))
            disp('Error : n should be inferior then N-1');  %Display error if n > N-1
            y3= 0;
    else
            s = zeros(N,1);  
            for i = n+1:N
                s(i) = 1 ;
            end 
            y3 = s;
            
            figure(2)
            subplot(3,1,1)
            stem(y3) ;  % Unit Step
            title('Step')  
            xlabel(' X')
            ylabel(' Y ')
           
    end
  
end