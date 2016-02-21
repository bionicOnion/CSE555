DO_TOY = true;
DO_BLEND = true;
DO_MIXED = true;
DO_COLOR2GRAY = true;

if DO_TOY 
    toyim = im2double(imread('./images/toy_problem.png'));
    im_out = toy_reconstruct(toyim);
    disp(['Error: ' num2str(sqrt(sum((toyim(:)-im_out(:)).^2)))])
end

if DO_BLEND
    im_background = imresize(im2double(imread('./images/im2.jpeg')), 0.25, 'bilinear');
    im_object = imresize(im2double(imread('./images/penguin-chick.jpeg')), 0.25, 'bilinear');

    % Compute the mask and alignment
    [im_s, mask_s] = alignSource(im_object, getMask(im_object), im_background);

    % Perform Poisson blending
    im_poisson_blend = poissonBlend(im_s, mask_s, im_background);
    figure(3), hold off, imshow(im_poisson_blend)
end

if DO_MIXED
    % If images haven't already been loaded, load them here
    if ~DO_BLEND
        im_background = imresize(im2double(imread('./images/im2.jpeg')), 0.25, 'bilinear');
        im_object = imresize(im2double(imread('./images/penguin-chick.jpeg')), 0.25, 'bilinear');
        [im_s, mask_s] = alignSource(im_object, getMask(im_object), im_background);
    end
    
    % Perform mixed-gradient blending
    im_mixed_blend = mixedBlend(im_s, mask_s, im_background);
    figure(4), hold off, imshow(im_mixed_blend);
end

if DO_COLOR2GRAY
    im_gr = color2gray(im2double(imread('./images/colorBlindTest35.png')));
    figure(5), hold off, imshow(im_gr)
end