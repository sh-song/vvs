clear;
clc;
close all;
run('/home/sh/Documents/matlab/vlfeat-0.9.21/toolbox/vl_setup')

addpath('./Givenfunctions');


%% 1. Load the input images (0pts)
im01 = imread('./Data/0001.JPG');
im02 = imread('./Data/0002.JPG');

%% 2. Extract features (2pts) 
% Plot im01
subplot(1,2,1);
imshow(im01, []);
title('im01');

% Extract SIFT features
im01 = rgb2gray(im01); 
im01 = single(im01);
[f1,d1] = vl_sift(im01); 

% Plot SIFT features
h = vl_plotframe(f1);
set(h,'color','b','linewidth',1);


% Plot im02
subplot(1,2,2);
imshow(im02, []);
title('im02');

% Extract SIFT features
im02 = rgb2gray(im02); 
im02 = single(im02);
[f2,d2] = vl_sift(im02);

% Plot SIFT features
h = vl_plotframe(f2);
set(h,'color','y','linewidth',1);


%% 3. Match features (3pts)

thresh = 10.0; 
[matches, scores] = vl_ubcmatch(d1, d2, thresh);
fprintf('Number of matching frames (features): %d\n', size(matches,2));

indices1 = matches(1,:); % Get matching features
f1match = f1(:,indices1);

indices2 = matches(2,:);
f2match = f2(:,indices2);


% Show matches
figure, imshow([im01,im02],[]);
o = size(im01, 2); % to plot on rightside
line([f1match(1,:);f2match(1,:)+o], ...
[f1match(2,:);f2match(2,:)]);

for i=1:size(f1match,2)
    x = f1match(1,i);
    y = f1match(2,i);
    text(x,y,sprintf('%d',i), 'Color', 'b');
end

for i=1:size(f2match,2)
    x = f2match(1,i);
    y = f2match(2,i);
    text(x+o,y,sprintf('%d',i), 'Color', 'y');
end

