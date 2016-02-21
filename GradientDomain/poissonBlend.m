function blended_img = poissonBlend(object, mask, target)
%poissonBlend Blend two images using the provided mask
%   object: the image fragment to be blended in
%   mask:   the logical mask of where in the target image the object should appear
%   target: the image into which the object will be blended

    % Obtain the indices of the source image and initialize a vector of indices into the image
    [height, width, channels] = size(target);
    pixelIndices = zeros(height, width); 
    pixelIndices(1:height*width) = 1:height*width;
    
    blended_img = zeros(size(target));
    
    % Initalize the constraint matrices (using sparse matrices since the vast majority of values
    %   in the matrices will be zeros)
    A = spalloc(size(find(mask),1) * 4, height * width, size(find(mask),1) * 4);
    b = zeros(size(find(mask),1) * 4, 1);
    
    for c = 1:channels
        % Initalize the equation counter to 0
        e = 0;
        
        % Set up the constraints to be solved
        for x = 2:width-1
            for y = 2:height-1
                % If within the confines of the mask, apply the blending constraints
                if mask(y,x)
                    % Left neighbor
                    e = e + 1;
                    A(e, pixelIndices(y,x)) = 1;
                    b(e) = object(y,x,c) - object(y,x-1,c);
                    if mask(y,x-1)
                        A(e, pixelIndices(y,x-1)) = -1;
                    else
                        b(e) = b(e) + target(y,x-1,c);
                    end
                    
                    % Right neighbor
                    e = e + 1;
                    A(e, pixelIndices(y,x)) = 1;
                    b(e) = object(y,x,c) - object(y,x+1,c);
                    if mask(y,x+1)
                        A(e, pixelIndices(y,x+1)) = -1;
                    else
                        b(e) = b(e) + target(y,x+1,c);
                    end
                    
                    % Top neighbor
                    e = e + 1;
                    A(e, pixelIndices(y,x)) = 1;
                    b(e) = object(y,x,c) - object(y-1,x,c);
                    if mask(y-1,x)
                        A(e, pixelIndices(y-1,x)) = -1;
                    else
                        b(e) = b(e) + target(y-1,x,c);
                    end
                    
                    % Bottom neighbor
                    e = e + 1;
                    A(e, pixelIndices(y,x)) = 1;
                    b(e) = object(y,x,c) - object(y+1,x,c);
                    if mask(y+1,x)
                        A(e, pixelIndices(y+1,x)) = -1;
                    else
                        b(e) = b(e) + target(y+1,x,c);
                    end
                end
            end
        end
        
        % Use the constraints above to solve for the blended image, reshaping it to match the
        %   dimensions of the source image
        blended_img(:,:,c) = full(reshape(A \ b, [height, width]));
    end
    
    % Copy the source image directly if outside of the blended region
    blended_img(~repmat(mask,[1,1,channels])) = target(~repmat(mask,[1,1,channels]));
end