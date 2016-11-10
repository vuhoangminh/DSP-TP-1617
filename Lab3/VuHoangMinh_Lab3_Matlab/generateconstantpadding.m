%=====================================================================
% Generate extended constant padding signal from a signal
%=====================================================================
function y = generateconstantpadding(x,numberPadding,valuePadding)
    m=length(x);
    n=m+2*numberPadding;
    y=zeros(1,n);
    for i=1:n
        y(i)=valuePadding;
    end
    y(numberPadding+1:numberPadding+m)=x;
end