function grayscale = color2gray(image)
%color2gray Convert RGB images to grayscale, preserving detail
%   Converts the provided image (which is simply assumed to be in the RGB color space; providing
%   different types of images will result in undefined behavior) to a false-intensity grayscale
%   representation using its HSV equivalent and mixed-gradient processing. The resulting image will
%   not be true to the original version's intensities (as rgb2gray is), but will instead preserve
%   more of the gradient and contrast of the image.
    
    image = rgb2hsv(image);
    
    [height, width, channels] = size(image);
    pixelIndices = zeros(height, width); 
    pixelIndices(1:height*width) = 1:height*width;
    
    A = sparse(height*width*4, height*width);
    b = zeros(height*width*4, 1);
    e = 0;
    
    for x = 1:width
        for y = 1:height
            % Left neighbor
            if x-1 > 0
                e = e + 1;
                satGrad = image(y,x,2) - image(y,x-1,2);
                valGrad = image(y,x,3) - image(y,x-1,3);
                b(e) = (abs(satGrad) > abs(valGrad))*satGrad + ...
                    (abs(satGrad) <= abs(valGrad))*valGrad;
                A(e, pixelIndices(y,x)) = 1;
                A(e, pixelIndices(y,x-1)) = -1;
            end
            
            % Right neighbor
            if x+1 <= width
                e = e + 1;
                satGrad = image(y,x,2) - image(y,x+1,2);
                valGrad = image(y,x,3) - image(y,x+1,3);
                b(e) = (abs(satGrad) > abs(valGrad))*satGrad + ...
                    (abs(satGrad) <= abs(valGrad))*valGrad;
                A(e, pixelIndices(y,x)) = 1;
                A(e, pixelIndices(y,x+1)) = -1;
            end
            
            % Top neighbor
            if y-1 > 0
                e = e + 1;
                satGrad = image(y,x,2) - image(y-1,x,2);
                valGrad = image(y,x,3) - image(y-1,x,3);
                b(e) = (abs(satGrad) > abs(valGrad))*satGrad + ...
                    (abs(satGrad) <= abs(valGrad))*valGrad;
                A(e, pixelIndices(y,x)) = 1;
                A(e, pixelIndices(y-1,x)) = -1;
            end
            
            % Bottom neighbor
            if y+1 <= height
                e = e + 1;
                satGrad = image(y,x,2) - image(y+1,x,2);
                valGrad = image(y,x,3) - image(y+1,x,3);
                b(e) = (abs(satGrad) > abs(valGrad))*satGrad + ...
                    (abs(satGrad) <= abs(valGrad))*valGrad;
                A(e, pixelIndices(y,x)) = 1;
                A(e, pixelIndices(y+1,x)) = -1;
            end
        end
    end
    
    % Solve for the correct values, reshape them into the correct dimensions, and normalize the
    %   resultant values
    grayscale = full(reshape(A \ b, [height, width]));
    grayscale = grayscale - min(min(grayscale));
    grayscale = grayscale / max(max(grayscale));
end