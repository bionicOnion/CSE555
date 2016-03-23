function synthesized = synthTexMinCut(srcTexture, patchDims, overlap, synthDims)
%synthTexMinCut Generate a new image with the same texture as some source image
%   Based on the source image in srcTexture, this function will generate a new image with the same
%     texture. Similar to synthesizeFromTexture, but with minimum-error boundary cuts.
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
            bestErrSurface = ones(patchDims);
            bestErr = Inf;
            bestPatch = [0, 0];
            for ii = 1:size(targetPatch,2):size(srcTexture, 2) - size(targetPatch,2)
                for jj = 1:size(targetPatch,1):size(srcTexture, 1) - size(targetPatch,1)
                    testPatch = srcTexture(jj:jj + size(targetPatch,1) - 1, ...
                        ii:ii + size(targetPatch,2) - 1, :).*(targetPatch ~= 0);
                    testErrSurface = (testPatch - targetPatch).^2;
                    testErr = sum(testErrSurface(:));
                    
                    if testErr < bestErr
                        bestErr = testErr;
                        bestErrSurface = testErrSurface;
                        bestPatch = [ii, jj];
                    end
                end
            end
            
            % Calculate the appropriate minimum-error cut mask
            width = overlap(1);
            if size(targetPatch,1) < overlap(1)
                width = size(targetPatch,1);
            end
            height = overlap(2);
            if size(targetPatch,2) < overlap(2)
                height = size(targetPatch,2);
            end
            vertMask = horzcat(minErrCut(rgb2gray(bestErrSurface(1:width,:,:))'), ...
                ones(size(targetPatch,2), size(targetPatch,1)-width))';
            horizMask = vertcat(minErrCut(rgb2gray(bestErrSurface(:,1:height,:)))', ...
                ones(size(targetPatch,2)-height, size(targetPatch,1)))';
            combinedMask = repmat(vertMask & horizMask, [1, 1, 3]);
            
            texPatch = srcTexture(bestPatch(2):bestPatch(2) + size(targetPatch,1) - 1, ...
                    bestPatch(1):bestPatch(1) + size(targetPatch,2) - 1, :);
                
            newPatch = combinedMask.*texPatch + (1 - combinedMask).*synthesized(j:j + ...
                size(targetPatch,1) - 1, i:i + size(targetPatch,2) - 1, :);
            
            synthesized(j:j + size(targetPatch,1) - 1, i:i + size(targetPatch,2) - 1, :) = newPatch;
        end
    end
end