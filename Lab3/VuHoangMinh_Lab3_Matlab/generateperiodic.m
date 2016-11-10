%=====================================================================
% Generate extended periodic signal from a signal
%=====================================================================
function y = generateperiodic(x,numberPadding)
    m=length(x);
    y=zeros(1,2*(m+numberPadding));
    y(numberPadding+1:numberPadding+m)=x;
    for i=1:m
        y(numberPadding+m+i)=x(i);
    end
end