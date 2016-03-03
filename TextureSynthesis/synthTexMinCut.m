function synthesized = synthTexMinCut(srcTexture, numPatches, overlap, synthDims)
%synthTexMinCut Generate a new image with the same texture as some source image
%   Based on the source image in srcTexture, this function will generate a new image with the same
%     texture. Similar to synthesizeFromTexture, but with minimum-error boundary cuts.
%
%   srcTexture: the image containing the texture to replicate
%   numPatches: the number of patches to be used for generation (in the form of a 2-element vector)
%   overlap:    by how many pixels the patches should overlap (another 2D vector)
%   synthDims:  the desired size of the output image

    synthesized = srcTexture;

end