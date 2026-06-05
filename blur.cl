__constant float gaussian[49] = {
    37, 49, 57, 61, 57, 49, 37,
    49, 64, 76, 80, 76, 64, 49,
    57, 76, 90, 95, 90, 76, 57,
    61, 80, 95,100, 95, 80, 61,
    57, 76, 90, 95, 90, 76, 57,
    49, 64, 76, 80, 76, 64, 49,
    37, 49, 57, 61, 57, 49, 37
};

__kernel void gaussian_blur(
    __global const uchar* input,
    __global uchar* output,
    int width,
    int height,
    int channels)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    for (int c = 0; c < channels; ++c)
    {
        float sum = 0.0f;

        for (int ky = -3; ky <= 3; ++ky)
        {
            for (int kx = -3; kx <= 3; ++kx)
            {
                int ix = clamp(x + kx, 0, width - 1);
                int iy = clamp(y + ky, 0, height - 1);

                float pixel = input[(iy * width + ix) * channels + c];
                float weight = gaussian[(ky + 3) * 7 + (kx + 3)];

                sum += pixel * weight;
            }
        }

        output[(y * width + x) * channels + c] = (uchar)(sum / 3264.0f);
    }
}
