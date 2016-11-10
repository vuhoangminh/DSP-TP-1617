function findtena (crossCorrelation, image, cmap)
% Sorting crossCorrelation and find 10 largest values
[maxArray,coordinateArray] = sort(crossCorrelation(:),'descend');
maxArray = maxArray(1:10);
coordinateArray = coordinateArray(1:10);
[xCoordinateArray,yCoordinateArray] = ind2sub(size(crossCorrelation),coordinateArray);

figure; imshow(crossCorrelation,[]);
title('Cross-correlation');
% Sorting crossCorrelation and find 10 largest values
drawfounda(image,cmap,xCoordinateArray,yCoordinateArray);
end

