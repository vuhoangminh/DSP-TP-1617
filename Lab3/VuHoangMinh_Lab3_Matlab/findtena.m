function findtena (crossCorrelation, image, cmap)
% Sorting crossCorrelation and find 10 largest values
nMax=10;
[maxArray,coordinateArray] = sort(crossCorrelation(:),'descend');
maxArray = maxArray(1:nMax);
coordinateArray = coordinateArray(1:nMax);
[xCoordinateArray,yCoordinateArray] = ind2sub(size(crossCorrelation),coordinateArray);

figure; imshow(crossCorrelation,[]);
title('Cross-correlation');
% Sorting crossCorrelation and find 10 largest values
drawfounda(image,cmap,xCoordinateArray,yCoordinateArray);
end

