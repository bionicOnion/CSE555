function [blendedImg] = blendImages(targetImage, sourceImage, mode)
%blendImages Automatically run through Poisson or mixed-gradient blending with the provided images
%   Based on the mode argument (either 'poisson' or 'mixed'), this function will perform image
%   blending using the specified method, returning the resulting composite image

    % If the specified mode is not recognized, return
    if ~strcmp(mode, 'poisson') && ~strcmp(mode, 'mixed')
       error('The mode %s is not recognized', mode); 
    end
    
    % Load the specified images
    source = imresize(im2double(imread(sourceImage)), 0.25, 'bilinear');
    target = imresize(im2double(imread(targetImage)), 0.25, 'bilinear');

    % Generate a user-specified blending mask
    mask = getMask(source);
    [source, mask] = alignSource(source, mask, target);
    
    % Perform blending using the specified method
    if strcmp(mode, 'poisson')
        blendedImg = poissonBlend(source, mask, target);
    elseif strcmp(mode, 'mixed')
        blendedImg = mixedBlend(source, mask, target);
    else
        blendedImg = target;
    end
end

