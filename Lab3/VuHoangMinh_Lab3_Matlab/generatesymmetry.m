%=====================================================================
% Generate extended symmetry signal from a signal
%=====================================================================
function y = generatesymmetry(x,numberPadding)
    m=length(x);
    y=zeros(1,2*(m+numberPadding));
    y(numberPadding+1:numberPadding+m)=x;
    for i=1:m
        y(numberPadding+m+i)=x(m-i+1);
    end
end