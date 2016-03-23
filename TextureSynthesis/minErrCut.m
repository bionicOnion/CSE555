function cutMask = minErrCut(errSurface)
%minErrCut Compute the minimum-error boundary cut

    % Fill in the minimum-error paths throughout the entire overlap region
    E = zeros(size(errSurface));
    E(1,:) = errSurface(1,:);
    for i = 2:size(errSurface,1)
        for j = 1:size(errSurface,2)
            minimum = E(i-1,j);
            if j > 1 && E(i-1,j-1) < minimum
                minimum = E(i-1,j-1);
            end
            if j < size(errSurface,2) && E(i-1, j+1) < minimum
                minimum = E(i-1, j+1);
            end
            E(i,j) = errSurface(i,j) + minimum;
        end
    end

    % Work back up the minimum-error path and generate a mask
    cutMask = zeros(size(errSurface));
    [~, cutLocation] = min(E(size(errSurface,1),:));
    cutMask(size(errSurface,1), cutLocation:size(errSurface,2)) = 1;
    for i = size(errSurface,1)-1:-1:1
        bestCutLoc = cutLocation;
        bestCutErr = E(i, bestCutLoc);
        if cutLocation > 1 && E(i, cutLocation - 1) < bestCutErr
            bestCutLoc = cutLocation - 1;
            bestCutErr = E(i, bestCutLoc);       
        end
        if cutLocation < size(errSurface,2) && E(i, cutLocation + 1) < bestCutErr
            bestCutLoc = cutLocation + 1;   
        end
        
        cutLocation = bestCutLoc;
        
        cutMask(i, cutLocation:size(errSurface,2)) = 1;
    end
end

