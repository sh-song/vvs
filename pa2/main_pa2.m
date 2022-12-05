clear;
clc;
close all;

addpath('./pfunctions');
load('./data/calib_result.mat');

im = imread('./data/plane_pattern.jpg');


disp('Undistort image');
roi = [740, 906, 558, 814];
chkim = zeros(size(im, 1), size(im, 2), 1);
chkim(roi(1):roi(2), roi(3):roi(4)) = 1;

rim = im(:, :, 1); gim = im(:, :, 2); bim = im(:, :, 3);

rim(chkim == 0) = 0;
gim(chkim == 0) = 0;
bim(chkim == 0) = 0;

tim(:, :, 1) = rim; 
tim(:, :, 2) = gim; 
tim(:, :, 3) = bim; 

disp('Detect checkerboard points');
imagePoints = detectCheckerboardPoints(tim);
imshow(im, 'InitialMagnification', 50); hold on;
plot(imagePoints(:, 1), imagePoints(:, 2), 'g*');
hold off;


%% Establish world X and image pixel x 
disp('Optimize camera0 pose');

baseline = 0:50:350;
world = [];
for x = baseline
    xs=ones(1, 8).*x;
    ys = baseline;
    world = [world, [xs;ys]]; % Concatenate on axis=1
end
world = [world; ones(1, size(world, 2))]; % add ones row, axis=0



plot3(world(1, :), world(2, :), world(3, :), '.');hold on;
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve nonlinear least squares problem for R, t
lb = [-pi,-pi,-pi,-1000,-1000,-1000]; %lower bound
ub = [pi,pi,pi,1000,1000,1000]; %upper bound
x0 = [-pi/2,0,0,0,0,1000]; % initial point
options=optimset('LargeScale','on','Display','iter','TolFun',1e-10,'TolX',1e-10,'MaxFunEvals',100000);

objective_func= @(x)RTfun(x, world, imagePoints(:,1:2)', IntParam0);
[x,resnorm,residual,exitflag,output] = lsqnonlin(objective_func, x0, lb, ub, options);

%% Evaluation Rt_p0 
Rt_p0=SetAxis(x); % make 4 by 4 Rt matrix in homogenous coordinate
disp('Rt');
disp(Rt_p0);

ws = [world(1:2,:); ... % world points
        zeros(1,size(world,2)); ... % originally z = 0
        ones(1,size(world,2))]; % added for H.C. calculation
        % 4 by 64 matrix

temp =Rt_p0*ws;


[original_xx,original_yy]=AddDistortion_fisheye(temp(1,:)./temp(3,:),...
                            temp(2,:)./temp(3,:),...
                            IntParam0(6:end), temp(3,:)<0);

[xx,yy]=my_AddDistortion_fisheye(temp(1,:)./temp(3,:),...
                            temp(2,:)./temp(3,:),...
                            IntParam0(6:end), temp(3,:)<0);


u=IntParam0(1)*xx+IntParam0(2)*yy+IntParam0(3);
v=IntParam0(4)*yy+IntParam0(5);



image_u = imagePoints(:, 1);
image_v = imagePoints(:, 2);
plot(image_u, image_v, 'b*'); hold on;
plot(u, v, 'r*', 'markersize', 6);
hold off;

n = size(u, 1);
error = norm([image_u - u, image_v - v], 2) / n;

fprintf('Reprojection Error ==%f.\n',error);

% Check my distortion function

n = size(xx, 2);
for i = 1:n
    diff_x = original_xx(1, i) - xx(1, i);
    diff_y = original_yy(1, i) - yy(1, i);


    fprintf('%dth x diff==%f.\n', i, diff_x);
    fprintf('%dth y diff==%f.\n', i, diff_y);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Prepare pixel points for AVM image

scale = 4;
tim_size = 3000;
wx = []; wy = [];
[wx, wy] = meshgrid(-(tim_size-1):scale:tim_size,...
                    -(tim_size-1):scale:tim_size);


cols = size(wx, 2); % 688
rows = size(wx, 1); % 775

wx = wx(:)'; % vector of indexes
wy = wy(:)';

plot(wx, wy, '.');
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

%% Top_view from camera front

temp = Rt_p0*[-wy;wx;zeros(1, length(wx)); ones(1, length(wx))];

[wxx,wyy]=my_AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam0(6:end),temp(3,:)<0);
u0=IntParam0(1)*wxx+IntParam0(2)*wyy+IntParam0(3) + 1;
v0=IntParam0(4)*wyy+IntParam0(5) + 1;

result0=my_Interpolation4_Color([u0;v0],double(im0));

for n=1:3
    top_view0(:,:,n)=reshape(result0(n,:),rows,cols);
end

top_view0(uint32(rows/2)-uint32(100/scale):end,:,:) = 0;
top_view0 = uint8(top_view0);
% figure(3); imshow(top_view0); drawnow;


%% Top_view from camera right

temp = CameraRelative01 * Rt_p0 *[-wy;wx;zeros(1, length(wx)); ones(1, length(wx))];

% Use the result of Q2.
[wxx,wyy]=my_AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam1(6:9),temp(3,:)<0);
u1=IntParam1(1)*wxx+IntParam1(2)*wyy+IntParam1(3) + 1;
v1=IntParam1(4)*wyy+IntParam1(5) + 1;

% Use the result of Q6
result1=Interpolation4_Color([u1;v1],double(im1));
for n=1:3
    top_view1(:,:,n)=reshape(result1(n,:),rows,cols);
end
top_view1(:,1:uint32(cols/2)+uint32(100/scale),:) = 0;
top_view1 = uint8(top_view1);
% figure; imshow(top_view1);

%% Top_view from camera back

temp = CameraRelative12 * CameraRelative01 * Rt_p0 * [-wy;wx;zeros(1, length(wx)); ones(1, length(wx))];

% Use the result of Q2.
[wxx,wyy]=my_AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam2(6:9),temp(3,:)<0);
u2=IntParam2(1)*wxx+IntParam2(2)*wyy+IntParam2(3) + 1;
v2=IntParam2(4)*wyy+IntParam2(5) + 1;

% Use the result of Q6
result1=Interpolation4_Color([u2;v2],double(im2));
for n=1:3
    top_view2(:,:,n)=reshape(result1(n,:),rows,cols);
end
top_view2(1:uint32(rows/2)+uint32(100/scale),:,:) = 0;
top_view2 = uint8(top_view2);
% figure; imshow(top_view2);



%% Top_view from camera left

temp = CameraRelative23* CameraRelative12 *CameraRelative01 * Rt_p0 *[-wy;wx;zeros(1, length(wx)); ones(1, length(wx))];

% Use the result of Q2.
[wxx,wyy]=my_AddDistortion_fisheye(temp(1,:)./temp(3,:),temp(2,:)./temp(3,:),IntParam3(6:9),temp(3,:)<0);
u3=IntParam3(1)*wxx+IntParam3(2)*wyy+IntParam3(3) + 1;
v3=IntParam3(4)*wyy+IntParam3(5) + 1;

% Use the result of Q6.
result1=Interpolation4_Color([u3;v3],double(im3));
for n=1:3
    top_view3(:,:,n)=reshape(result1(n,:),rows,cols);
end
top_view3(:,uint32(cols/2)-uint32(100/scale):end,:) = 0;
top_view3 = uint8(top_view3);
% figure; imshow(top_view3);


%% Four IPM images into One All Around View Image (Extra)

total_view = top_view0;
top_view = top_view0;
chan0 = top_view(:, :, 1);
chan1 = top_view(:, :, 2);
chan2 = top_view(:, :, 3);
 
for u=1:size(total_view, 1);
    for v = 1:size(total_view, 2);
        
        flag0 = not (chan0(u, v) == 0);
        flag1 = not (chan1(u, v) == 0);
        flag2 = not (chan2(u, v) == 0);
        if flag0 && flag1 && flag2
 
            total_view(u, v, 1) = chan0(u, v); 
            total_view(u, v, 2) = chan1(u, v); 
            total_view(u, v, 3) = chan2(u, v); 
        end
    end
end
    

top_view = top_view1;
chan0 = top_view(:, :, 1);
chan1 = top_view(:, :, 2);
chan2 = top_view(:, :, 3);
 
for u=1:size(total_view, 1);
    for v = 1:size(total_view, 2);
        
        flag0 = not (chan0(u, v) == 0);
        flag1 = not (chan1(u, v) == 0);
        flag2 = not (chan2(u, v) == 0);
        if flag0 && flag1 && flag2
 
            total_view(u, v, 1) = chan0(u, v); 
            total_view(u, v, 2) = chan1(u, v); 
            total_view(u, v, 3) = chan2(u, v); 
        end
    end
end
    
top_view = top_view2;
chan0 = top_view(:, :, 1);
chan1 = top_view(:, :, 2);
chan2 = top_view(:, :, 3);
 
for u=1:size(total_view, 1);
    for v = 1:size(total_view, 2);
        
        flag0 = not (chan0(u, v) == 0);
        flag1 = not (chan1(u, v) == 0);
        flag2 = not (chan2(u, v) == 0);
        if flag0 && flag1 && flag2
 
            total_view(u, v, 1) = chan0(u, v); 
            total_view(u, v, 2) = chan1(u, v); 
            total_view(u, v, 3) = chan2(u, v); 
        end
    end
end

top_view = top_view3;
chan0 = top_view(:, :, 1);
chan1 = top_view(:, :, 2);
chan2 = top_view(:, :, 3);
 
for u=1:size(total_view, 1);
    for v = 1:size(total_view, 2);
        
        flag0 = not (chan0(u, v) == 0);
        flag1 = not (chan1(u, v) == 0);
        flag2 = not (chan2(u, v) == 0);
        if flag0 && flag1 && flag2
 
            total_view(u, v, 1) = chan0(u, v); 
            total_view(u, v, 2) = chan1(u, v); 
            total_view(u, v, 3) = chan2(u, v); 
        end
    end
end
figure; imshow(total_view);