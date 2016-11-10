%=====================================================================
% Compute convolution
%=====================================================================
function outImage = convolution2D(inImage, K)
% Find size of images
[r, c] = size(inImage);         
[m, n] = size(K);

% Find center, left, right, top and bottom of kernel
center = floor((size(K)+1)/2);
left = center(2) - 1;
right = n - center(2);
top = center(1) - 1;
bottom = m - center(1);

% Declare padded matrix from inImage and kernel K
Rep = zeros(r + top + bottom, c + left + right);

% Copy inImage to padded matrix
for x = 1 + top : r + top
    for y = 1 + left : c + left
        Rep(x,y) = inImage(x - top, y - left);
    end
end

% Declare convolved image
outImage = zeros(r , c);

% Compute convolution
for x = 1 : r
    for y = 1 : c
        for i = 1 : m
            for j = 1 : n
                q = x - 1;
                w = y - 1;
                outImage(x, y) = outImage(x, y) + (Rep(i + q, j + w) * K(i, j));
            end
        end
    end
end
end    