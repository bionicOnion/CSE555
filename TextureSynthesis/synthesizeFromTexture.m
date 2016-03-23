function synthesized = synthesizeFromTexture(srcTexture, patchDims, overlap, synthDims)
%synthesizeFromTexture Generate a new image with the same texture as some source image
%   Based on the source image in srcTexture, this function will generate a new image with the same
%     texture.
%
%   srcTexture: the image containing the texture to replicate
%   patchDims: the desired dimensions for patch sampling (as a 2D vector)
%   overlap:    by how many pixels the patches should overlap (another 2D vector)
%   synthDims:  the desired size of the output image

    synthesized = zeros(synthDims);
    
    for i = 1:patchDims(2) - overlap(2):synthDims(2)
        for j = 1:patchDims(1) - overlap(1):synthDims(1)
            
            % Compute patch boundaries
            xBound = i + patchDims(2) - 1;
            if xBound > synthDims(2)
                xBound = synthDims(2);
            end
            yBound = j + patchDims(1) - 1;
            if yBound > synthDims(1)
                yBound = synthDims(1);
            end
            
            % Find the target region in the synthesized image
            targetPatch = synthesized(j:yBound, i:xBound, :);
            
            % Find the minimum-error patch
            bestErr = Inf;
            bestPatch = [0, 0];
            for ii = 1:size(targetPatch,2):size(srcTexture, 2) - size(targetPatch,2)
                for jj = 1:size(targetPatch,1):size(srcTexture, 1) - size(targetPatch,1)
                    testPatch = srcTexture(jj:jj + size(targetPatch,1) - 1, ...
                        ii:ii + size(targetPatch,2) - 1, :);
                    testPatch = testPatch.*(targetPatch ~= 0);
                    testErrSurface = (testPatch - targetPatch).^2;
                    testErr = sum(testErrSurface(:));
                    
                    if testErr < bestErr
                        bestErr = testErr;
                        bestPatch = [ii, jj];
                    end
                end
            end
            
            % Insert the best-matching patch into the generated image
            synthesized(j:j + size(targetPatch,1) - 1, i:i + size(targetPatch,2) - 1, :) = ...
                srcTexture(bestPatch(2):bestPatch(2) + size(targetPatch,1) - 1, ...
                    bestPatch(1):bestPatch(1) + size(targetPatch,2) - 1, :);
        end
    end
end