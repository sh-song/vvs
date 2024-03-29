clear;
clc;
close all;
addpath('./pfunctions');
load('./data/calib_result.mat');

%% Load Checkerboard
im = imread('./data/plane_pattern.jpg');

% Undistort image
disp('Undistort image');

roi = [740,906,558,814];
chkim = zeros(size(im,1), size(im,2), 1);
chkim(roi(1):roi(2),roi(3):roi(4)) = 1;
rim = im(:,:,1); gim = im(:,:,2); bim = im(:,:,3);
rim(chkim == 0) = 0;
gim(chkim == 0) = 0;
bim(chkim == 0) = 0;
tim(:,:,1) = rim; tim(:,:,2) = gim; tim(:,:,3) = bim;

%% Detect Checkerboard Points 
disp('Detect checkerboard points');
imagePoints = detectCheckerboardPoints(tim);
figure,
imshow(im, 'InitialMagnification', 50); hold on;
plot(imagePoints(:, 1), imagePoints(:, 2), '*-g');
hold off;

%% Establish world X and image pixel x 
disp('Optimize camera0 pose');
yinc = repmat([50,0], 8,1);

% Q1. Insert Your Code Here! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO:  Establish world X and image pixel x
% INPUT: Your define coordinate
% OUTPUT: Homogeneous world coordinate (variable: world)
% [10 Point]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute Relative Pose (Rt_p0) btw, Cam0 and World origin
lb = [-pi,-pi,-pi,-1000,-1000,-1000];
ub = [pi,pi,pi,1000,1000,1000];
x0 = [-pi/2,0,0,0,0,1000];
options=optimset('LargeScale','on','Display','iter','TolFun',1e-10,'TolX',1e-10,'MaxFunEvals',100000);
[x,resnorm,residual,exitflag,output] = lsqnonlin(@(x)RTfun(x, world, imagePoints(:,1:2)', IntParam0), ...
                                                 x0, lb, ub, options);

%% Evaluation Rt_p0 
Rt_p0=SetAxis(x);
temp=SetAxis(x)*[world(1:2,:);zeros(1,size(world,2));ones(1,size(world,2))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q2. Implement Your Code Here ! 
% TODO: Add distortion to world points.
% [10 Point]
% INPUT and OUTPUT : see the following AddDistortion_fisheye
% Distortion model Please see. https://euratom-software.github.io/calcam/html/intro_theory.html#fisheye-lens-distirtion-model

% Please make your own AddDistortion_fisheye function
% [input]
% temp: 4x64, 4x(# of points)
% [output]
% xx: 1x64, 1x(# of points)
% yy: 1x64, 1x(# of points)

[xx,yy]=AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam0(6:end),temp(3,:)<0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u=IntParam0(1)*xx+IntParam0(2)*yy+IntParam0(3);
v=IntParam0(4)*yy+IntParam0(5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q3. Implement Your Code Here ! 
% TODO : Display Reprojection Error with Pixel Error Metric
% INPUT : reprojected point (variable: u,v), and detected point (variable: imagePoints)
% [10 Point]

% Plot imagePoints and u,v for comparison


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Prepare pixel points for AVM image

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q4. Implement Your Code Here ! 
% TODO : Create Your Own Image Coordinate to make a IPM image
% INPUT : Your Define Image Coordinate
% OUTPUT : Pixel Position (wx, wy) 
% [10 Point]

% [Input]
% Define your own image size for top-view (ex, 2,000*2,000)
% [Output]
% wx, 1x(2,000*2,000) 1x(# of pixels)
% wy, 1x(2,000*2,000) 1x(# of pixels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Linear blend boundary
im0 = imread('./data/cam_f.jpg');
im1 = imread('./data/cam_r.jpg');
im2 = imread('./data/cam_b.jpg');
im3 = imread('./data/cam_l.jpg'); 
figure, 
subplot(2,2,1); imshow(im0);
subplot(2,2,2); imshow(im1);
subplot(2,2,3); imshow(im2);
subplot(2,2,4); imshow(im3);

%% Top_view from camera camera

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q5. Implement Your Code Here ! 
% TODO : Compute Relative Pose btw. world and front camera
% INPUT : wx, wy
% OUTPUT : 2D position(variable: temp) in camera coordinate corresponding to (wx, wy) 
% [2.5 Point]

% HINT: Make wx, wy to homogeneous coordindate system
% [Input]
% wx, 1x(2,000*2,000) 1x(# of pixels)
% wy, 1x(2,000*2,000) 1x(# of pixels)
% [Output]
% temp, 4x(2,000*2,000) 4x(# of pixels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the result of Q2.
[wxx,wyy]=AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam0(6:end),temp(3,:)<0);
u0=IntParam0(1)*wxx+IntParam0(2)*wyy+IntParam0(3) + 1;
v0=IntParam0(4)*wyy+IntParam0(5) + 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q6. Implement Your Code Here ! 
% TODO: Image Generation by Backward warping.
% INPUT and OUTPUT : Please see the following Interpolation4_Color() function
% [10 Point]

% Please make your own Interpolation4_Color function
% [Input]
% u0, 1x(2,000*2,000) 1x(# of pixels)
% v0, 1x(2,000*2,000) 1x(# of pixels)
% [Output]
% result0, 3x(2,000*2,000) (color_channel)x(# of pixels)
result0=Interpolation4_Color([u0;v0],double(im0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n=1:3
    top_view0(:,:,n)=reshape(result0(n,:),rows,cols);
end
top_view0(uint32(rows/2)-uint32(150/scale):end,:,:) = 0;
top_view0 = uint8(top_view0);
figure(3); imshow(top_view0); drawnow;



%% Top_view from camera right

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q7. Implement Your Code Here ! 
% TODO : Compute Relative Pose btw. world and right camera
% INPUT : wx, wy
% OUTPUT : 2D position(variable: temp) in camera coordinate corresponding to (wx, wy) 
% [2.5 Point]

% [Input]
% wx, 1x(2,000*2,000) 1x(# of pixels)
% wy, 1x(2,000*2,000) 1x(# of pixels)
% [Output]
% temp, 4x(2,000*2,000) 4x(# of pixels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the result of Q2.
[wxx,wyy]=AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam1(6:9),temp(3,:)<0);
u1=IntParam1(1)*wxx+IntParam1(2)*wyy+IntParam1(3) + 1;
v1=IntParam1(4)*wyy+IntParam1(5) + 1;

% Use the result of Q6
result1=Interpolation4_Color([u1;v1],double(im1));
for n=1:3
    top_view1(:,:,n)=reshape(result1(n,:),rows,cols);
end
top_view1(:,1:uint32(cols/2)+uint32(100/scale),:) = 0;
top_view1 = uint8(top_view1);
figure; imshow(top_view1);


%% Top_view from camera back

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q8. Implement Your Code Here ! 
% TODO : Compute Relative Pose btw. world and back camera
% INPUT : wx, wy
% OUTPUT : 2D position(variable: temp) in camera coordinate corresponding to (wx, wy) 
% [2.5 Point]

% [Input]
% wx, 1x(2,000*2,000) 1x(# of pixels)
% wy, 1x(2,000*2,000) 1x(# of pixels)
% [Output]
% temp, 4x(2,000*2,000) 4x(# of pixels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the result of Q2.
[wxx,wyy]=AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam2(6:9),temp(3,:)<0);
u2=IntParam2(1)*wxx+IntParam2(2)*wyy+IntParam2(3) + 1;
v2=IntParam2(4)*wyy+IntParam2(5) + 1;

% Use the result of Q6
result1=Interpolation4_Color([u2;v2],double(im2));
for n=1:3
    top_view2(:,:,n)=reshape(result1(n,:),rows,cols);
end
top_view2(1:uint32(rows/2)+uint32(100/scale),:,:) = 0;
top_view2 = uint8(top_view2);
figure; imshow(top_view2);


%% Top_view from camera left

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q9. Implement Your Code Here ! 
% TODO : Compute Relative Pose btw. world and left camera
% INPUT : wx, wy
% OUTPUT : 2D position(variable: temp) in camera coordinate corresponding to (wx, wy) 
% [2.5 Point]


% [Input]
% wx, 1x(2,000*2,000) 1x(# of pixels)
% wy, 1x(2,000*2,000) 1x(# of pixels)
% [Output]
% temp, 4x(2,000*2,000) 4x(# of pixels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the result of Q2.
[wxx,wyy]=AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam3(6:9),temp(3,:)<0);
u3=IntParam3(1)*wxx+IntParam3(2)*wyy+IntParam3(3) + 1;
v3=IntParam3(4)*wyy+IntParam3(5) + 1;

% Use the result of Q6.
result1=Interpolation4_Color([u3;v3],double(im3));
for n=1:3
    top_view3(:,:,n)=reshape(result1(n,:),rows,cols);
end
top_view3(:,uint32(cols/2)-uint32(100/scale):end,:) = 0;
top_view3 = uint8(top_view3);
figure; imshow(top_view3);





