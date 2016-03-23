function synthesized = textureTransfer(textureSrc, contentSrc, patchDims, overlap, correspondence)
%textureTransfer Recreate the content image with the texture of the other image

    synthesized = zeros(size(contentSrc));
    
    for i = 1:patchDims(2) - overlap(2):size(synthesized,2)
        for j = 1:patchDims(1) - overlap(1):size(synthesized,1)
            
            % Compute patch boundaries
            xBound = i + patchDims(2) - 1;
            if xBound > size(synthesized,2);
                xBound = size(synthesized,2);
            end
            yBound = j + patchDims(1) - 1;
            if yBound > size(synthesized,1)
                yBound = size(synthesized,1);
            end
            
            % Find the target region in the synthesized image
            targetPatch = synthesized(j:yBound, i:xBound, :);
            contentPatch = contentSrc(j:yBound, i:xBound, :);
            
            % Find the minimum-error patch
            bestErr = Inf;
            bestPatch = [0, 0];
            for ii = 1:size(targetPatch,2):size(textureSrc, 2) - size(targetPatch,2)
                for jj = 1:size(targetPatch,1):size(textureSrc, 1) - size(targetPatch,1)
                    testPatch = textureSrc(jj:jj + size(targetPatch,1) - 1, ...
                        ii:ii + size(targetPatch,2) - 1, :).*(targetPatch ~= 0);
                    
                    % Compute the overlap error
                    testErrSurface = (testPatch - targetPatch).^2;
                    overlapErr = sum(testErrSurface(:));
                    
                    % Compute the correspondence error
                    correspondErrSurface = ...
                        (imgaussfilt(testPatch, 2) - imgaussfilt(contentPatch, 2)).^2;
                    correspondErr = sum(correspondErrSurface(:));
                    
                    % Combine the two error terms based on the provided correspondence value
                    testErr = correspondence*overlapErr + (1 - correspondence)*correspondErr;
                    
                    % If this patch error is the best found thus far, record it
                    if testErr < bestErr
                        bestErr = testErr;
                        bestErrSurface = testErrSurface;
                        bestPatch = [ii, jj];
                    end
                end
            end
            
            % Calculate the appropriate minimum-error cut mask
            vertMask = horzcat(minErrCut(rgb2gray(bestErrSurface(1:overlap(1),:,:))'), ...
                ones(size(targetPatch,2), size(targetPatch,1)-overlap(1)))';
            horizMask = vertcat(minErrCut(rgb2gray(bestErrSurface(:,1:overlap(2),:)))', ...
                ones(size(targetPatch,2)-overlap(2), size(targetPatch,1)))';
            combinedMask = repmat(vertMask & horizMask, [1, 1, 3]);
            
            % Insert the best-matching patch into the generated image
            texPatch = textureSrc(bestPatch(2):bestPatch(2) + size(targetPatch,1) - 1, ...
                bestPatch(1):bestPatch(1) + size(targetPatch,2) - 1, :);
            newPatch = combinedMask.*texPatch + (1 - combinedMask).*synthesized(j:j + ...
                size(targetPatch,1) - 1, i:i + size(targetPatch,2) - 1, :);
            synthesized(j:j + size(targetPatch,1) - 1, i:i + size(targetPatch,2) - 1, :) = newPatch;
        end
    end
end