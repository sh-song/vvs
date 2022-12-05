function [xx, yy] = my_AddDistortion_fisheye(x_w, y_w, dist, z_mask)
    input = [x_w; y_w]';

    r_vec = vecnorm(input, 2, 2); % L2 norm
    theta_vec = atan(r_vec);

    n = size(r_vec);
    xx = zeros(n)';
    yy = zeros(n)';

    for i = 1:size(r_vec)
        if z_mask(i) == 1
            xx(i) = 999;
            yy(i) = 999;
        else
            th = theta_vec(i);
            model = (1 + dist(1)*th^2 + dist(2)*th^4 + dist(3)*th^6 + dist(4)*th^8)*th / r_vec(i);
            xx(i) = model * input(i, 1);
            yy(i) = model * input(i, 2);
    end
end