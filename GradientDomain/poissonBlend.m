function [ blended_img ] = poissonBlend(fg_img, mask, bg_img)
%poissonBlend Blend two images using the provided mask
%   TODO

    % Obtain the indices of the source image and initialize a vector of indices into the image
    [height, width, channels] = size(bg_img);
    pixelIndices = zeros(height, width); 
    pixelIndices(1:height*width) = 1:height*width;
    
    blended_img = zeros(size(bg_img));
    
    for c = 1:channels
        % Initalize the equation counter to 0
        e = 1;
        
        % Initalize the constraint matrices (using sparse matrices since the vast majority of values
        %   in the matrices will be zeros)
        A = sparse(height * width * 8, height * width);
        b = sparse(height * width * 8, 1);
        
        % Set up the constraints to be solved
        for x = 2:width-1
            for y = 2:height-1
                % Constraint 1: Similarity with source
                A(e, pixelIndices(y,x)) = 1;
                A(e, pixelIndices(y,x-1)) = -mask(y,x-1);
                b(e) = bg_img(y,x,c) - bg_img(y,x-1,c);
                
                A(e+1, pixelIndices(y,x)) = 1;
                A(e+1, pixelIndices(y,x+1)) = -mask(y,x+1);
                b(e+1) = bg_img(y,x,c) - bg_img(y,x+1,c);
                
                A(e+2, pixelIndices(y,x)) = 1;
                A(e+2, pixelIndices(y-1,x)) = -mask(y-1,x);
                b(e+2) = bg_img(y,x,c) - bg_img(y-1,x,c);
                
                A(e+3, pixelIndices(y,x)) = 1;
                A(e+3, pixelIndices(y+1,x)) = -mask(y+1,x);
                b(e+3) = bg_img(y,x,c) - bg_img(y+1,x,c);
                
                % Constraint 2: Similarity with target
                A(e+4, pixelIndices(y,x)) = 1;
                b(e+4) = bg_img(y,x,c) - bg_img(y,x-1,c) + mask(y,x)*fg_img(y,x-1,c);
                
                A(e+5, pixelIndices(y,x)) = 1;
                b(e+5) = bg_img(y,x,c) - bg_img(y,x+1,c) + mask(y,x)*fg_img(y,x+1,c);
                
                A(e+6, pixelIndices(y,x)) = 1;
                b(e+6) = bg_img(y,x,c) - bg_img(y-1,x,c) + mask(y,x)*fg_img(y-1,x,c);
                
                A(e+7, pixelIndices(y,x)) = 1;
                b(e+7) = bg_img(y,x,c) - bg_img(y+1,x,c) + mask(y,x)*fg_img(y+1,x,c);
            end
            e = e + 8;
        end
        
        % Use the constraints above to solve for the blended image, reshaping it to match the
        %   dimensions of the source image
        blended_img(:,:,c) = full(reshape(A \ b, [height, width]));
    end
end

