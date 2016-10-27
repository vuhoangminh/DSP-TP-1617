function y1 =  Dirac(n,N) %Function Defination 

    if ((n<1)||(n>N))
            disp('Error : n should be inferior then N-1');  %Display error if n > N-1
            y1= 0;
    else
            s = zeros(1,N);  
            s(n) = 1 ;
            y1 = s;
           
    end
  
end