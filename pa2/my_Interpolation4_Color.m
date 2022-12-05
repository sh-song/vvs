function result = my_Interpolation4_Color(uv, im)

    len = size(uv, 2)
    h = size(im, 1);
    w = size(im, 2);

    us = uv(1, :);
    vs = uv(2, :);

    rows = sqrt(len);
    cols = sqrt(len);
    result = zeros(3, len);
    for n=1:3
        chan = im(:, :, n);
        for i=1:len
            v = int32(us(i));
            u = int32(vs(i));

            up = [u-1, v];
            down = [u+1, v];
            left = [u, v-1];
            right = [u, v+1];

            if (u > 0) && (u < h) && (v > 0) && (v < w)
                sum = chan(up(1), up(2)) ...
                    + chan(down(1), down(2)) ...
                    + chan(left(1), left(2)) ...
                    + chan(right(1), right(2));
                result(n, i) = sum / 4; % avg
                % result(n, i) = chan(u, v);
            else
                result(n, i) = 0;
        end
    end
end