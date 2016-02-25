function [im_blended, im_aligned] = performBlending(im_object, im_target, mode)
%performBlending Blend the provided images using the blending model specified by the mode parameter
%   Perform automated blending of two provided images using either Poisson on mixed-gradient
%     blending methods

    % If the mode is unrecognized, throw an error
    if ~strcmp(mode, 'poisson') && ~strcmp(mode, 'mixed')
        error('Unrecognized blending mode: %s', mode);
    end

    % Generate a mask and align the images
    [source, mask] = alignSource(im_object, getMask(im_object), im_target);
    channels = size(im_target,3);
    im_aligned(repmat(mask,[1,1,channels])) = source(repmat(mask,[1,1,channels]));
    im_aligned(~repmat(mask,[1,1,channels])) = im_target(~repmat(mask,[1,1,channels]));
    im_aligned = reshape(im_aligned, size(im_target));

    % Compute the blended image
    if strcmp(mode, 'poisson')
        im_blended = poissonBlend(source, mask, im_target);
    elseif strcmp(mode, 'mixed')
        im_blended = mixedBlend(source, mask, im_target);
    end
end