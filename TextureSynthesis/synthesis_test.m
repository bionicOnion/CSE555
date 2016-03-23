% Read in sample textures
water = im2double(imread('textures/water.png'));
leaf = im2double(imread('textures/leaf.jpeg'));
farm = im2double(imread('textures/farm.jpeg'));

% Generate synthesized versions of the textures without minimum-error cutting
disp('Creating tiled synthetic textures...');
waterSynth = synthesizeFromTexture(water, [50, 50], [5, 5], [400, 400, 3]);
leafSynth = synthesizeFromTexture(leaf, [50, 50], [5, 5], [400, 400, 3]);
farmSynth = synthesizeFromTexture(farm, [50, 50], [5, 5], [400, 400, 3]);

% Generate synthesized versions of the textures WITH minimum-error cutting
disp('Creating minimum-error cut synthetic textures...');
waterSynthMinCut = synthTexMinCut(water, [50, 50], [5, 5], [400, 400, 3]);
leafSynthMinCut = synthTexMinCut(leaf, [50, 50], [5, 5], [400, 400, 3]);
farmSynthMinCut = synthTexMinCut(farm, [50, 50], [5, 5], [400, 400, 3]);

% Display the synthetic images
figure(1), subplot(1,3,1), imshow(water), ...
    subplot(1,3,2), imshow(waterSynth), ...
    subplot(1,3,3), imshow(waterSynthMinCut);
figure(2), subplot(1,3,1), imshow(leaf), ...
    subplot(1,3,2), imshow(leafSynth), ...
    subplot(1,3,3), imshow(leafSynthMinCut);
figure(3), subplot(1,3,1), imshow(farm), ...
    subplot(1,3,2), imshow(farmSynth), ...
    subplot(1,3,3), imshow(farmSynthMinCut);