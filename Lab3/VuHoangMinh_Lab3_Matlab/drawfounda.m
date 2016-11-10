%=====================================================================
% Draw red squares around found A
%=====================================================================
function drawfounda(image,cmap_text,I,J)
    n=size(I);
    figure;
    imshow(image,cmap_text);
    axis image off;
    colormap gray;
    title('Found first 10 a');
    for i=1:n
        hold on
        x=I(i)-8-8;
        X=I(i)+8-8;
        y=J(i)-8-8;
        Y=J(i)+8-8;
        plot([y y Y Y y],[x X X x x],'r')
    end
end

