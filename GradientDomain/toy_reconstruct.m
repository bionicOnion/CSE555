function [reconstructed] = toy_reconstruct(image)
%toy_reconstruct Reconstruct an image from its gradient
%   After converting the original image (provided as an argument) into the
%     gradient domain, this function reconstructs the image (using a single
%     pixel as a point of reference) and returns it to the function's
%     caller

    % Obtain the indices of the source image and initialize a vector of
    %   indices into the image
    [height, width] = size(image); 
    pixelIndices = zeros(height, width); 
    pixelIndices(1:height*width) = 1:height*width;
    
    % Initialize the equation counter to 0
    e = 0;
    
    % Initalize the constraint matrices (using sparse matrices since the
    %   vast majority of values in the matrices will be zeros)
    A = sparse(height * width * 2 + 1, height * width);
    b = sparse(height * width * 2 + 1, 1);
    
    % Set up the first constraint: equivalence of the horizontal components
    %   of the gradients
    for x = 1:width-1
        for y = 1:height
            e = e + 1;
            A(e, pixelIndices(y,x+1)) = 1;
            A(e, pixelIndices(y,x)) = -1;
            b(e) = image(y,x+1) - image(y,x);
        end
    end
    
    % Set up the second constraint: equivalence of the vertical components
    %   of the gradients
    for x = 1:width
        for y = 1:height-1
            e = e + 1;
            A(e, pixelIndices(y+1,x)) = 1;
            A(e, pixelIndices(y,x)) = -1;
            b(e) = image(y+1,x) - image(y,x);
        end
    end

    % Set up the third constraint: the top left pixel of the reconstructed
    %   image should have the same intensity as the pixel at the same
    %   location in the source image
    e = e + 1
    A(e, pixelIndices(1,1)) = 1;
    b(e) = image(1,1);

    % Use the constraints above to solve for the reconstructed image,
    %   reshaping it to match the dimensions of the source image
    reconstructed = full(reshape(lscov(A, b), [width, height]));
end

