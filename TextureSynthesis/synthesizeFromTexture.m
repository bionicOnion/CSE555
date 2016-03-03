function synthesized = synthesizeFromTexture(srcTexture, numPatches, overlap, synthDims)
%synthesizeFromTexture Generate a new image with the same texture as some source image
%   Based on the source image in srcTexture, this function will generate a new image with the same
%     texture.
%
%   srcTexture: the image containing the texture to replicate
%   numPatches: the number of patches to be used for generation (in the form of a 2-element vector)
%   overlap:    by how many pixels the patches should overlap (another 2D vector)
%   synthDims:  the desired size of the output image

    synthesized = srcTexture;
    
    % Copy a random patch into the top left corner of the image
    % For every other patch in the image, compute the SSD of the overlap with its neighbors
    % Use Poisson blending to insert the new patch into the correct poisition

end