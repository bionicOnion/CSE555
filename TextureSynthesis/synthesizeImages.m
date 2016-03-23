function synthesizeImages(srcTexture)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    synthDims = [size(srcTexture,1)*2, size(srcTexture,2)*2, 3];

    blockBased = synthesizeFromTexture(srcTexture, [100, 100], [10, 10], synthDims);
    minCutBased = synthTexMinCut(srcTexture,[100, 100], [10, 10], synthDims);
    
    subplot(1,2,1), imshow(blockBased), subplot(1,2,2), imshow(minCutBased);
    
    imwrite(blockBased, strcat('./output/', inputname(1), 'Block.jpeg'));
    imwrite(minCutBased, strcat('./output/', inputname(1), 'MinCut.jpeg'));
end

