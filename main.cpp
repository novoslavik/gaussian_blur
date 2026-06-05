#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include <CL/opencl.hpp>

static const float gaussian[49] = {
    37, 49, 57, 61, 57, 49, 37,
    49, 64, 76, 80, 76, 64, 49,
    57, 76, 90, 95, 90, 76, 57,
    61, 80, 95,100, 95, 80, 61,
    57, 76, 90, 95, 90, 76, 57,
    49, 64, 76, 80, 76, 64, 49,
    37, 49, 57, 61, 57, 49, 37
};

cv::Mat blur_cpu(const cv::Mat& img) {
    const int width    = img.cols;
    const int height   = img.rows;
    const int channels = img.channels();
    cv::Mat result(height, width, img.type());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int ky = -3; ky <= 3; ++ky) {
                    for (int kx = -3; kx <= 3; ++kx) {
                        const int ix = std::clamp(x + kx, 0, width - 1);
                        const int iy = std::clamp(y + ky, 0, height - 1);
                        sum += img.data[(iy * width + ix) * channels + c]
                             * gaussian[(ky + 3) * 7 + (kx + 3)];
                    }
                }
                result.data[(y * width + x) * channels + c] = static_cast<uchar>(sum / 3264.0f);
            }
        }
    }
    return result;
}

cv::Mat blur_cpu_mt(const cv::Mat& img, int num_threads = 0) {
    if (num_threads <= 0)
        num_threads = static_cast<int>(std::thread::hardware_concurrency());

    const int width    = img.cols;
    const int height   = img.rows;
    const int channels = img.channels();
    cv::Mat result(height, width, img.type());

    auto worker = [&](int row_start, int row_end) {
        for (int y = row_start; y < row_end; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < channels; ++c) {
                    float sum = 0.0f;
                    for (int ky = -3; ky <= 3; ++ky) {
                        for (int kx = -3; kx <= 3; ++kx) {
                            const int ix = std::clamp(x + kx, 0, width - 1);
                            const int iy = std::clamp(y + ky, 0, height - 1);
                            sum += img.data[(iy * width + ix) * channels + c]
                                 * gaussian[(ky + 3) * 7 + (kx + 3)];
                        }
                    }
                    result.data[(y * width + x) * channels + c] = static_cast<uchar>(sum / 3264.0f);
                }
            }
        }
    };

    const int rows_per_thread = height / num_threads;
    std::vector<std::thread> threads(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        const int row_start = i * rows_per_thread;
        const int row_end   = (i == num_threads - 1) ? height : row_start + rows_per_thread;
        threads[i] = std::thread(worker, row_start, row_end);
    }
    for (auto& t : threads)
        t.join();

    return result;
}

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    const cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    const cl::Device device = devices[0];

    std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    assert(device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);

    const cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::ifstream file("blur.cl");
    const std::string source(std::istreambuf_iterator<char>(file), {});

    const cl::Program program(context, source);
    program.build({device});

    cl::Kernel kernel(program, "gaussian_blur");

    cv::Mat img = cv::imread("input.png", cv::IMREAD_COLOR);

    const int    width    = img.cols;
    const int    height   = img.rows;
    const int    channels = img.channels();
    const size_t size     = width * height * channels;

    // --- GPU ---
    cl::Buffer inputBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, img.data);
    cl::Buffer outputBuf(context, CL_MEM_WRITE_ONLY, size);

    kernel.setArg(0, inputBuf);
    kernel.setArg(1, outputBuf);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, channels);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange, nullptr, &event);
    event.wait();

    const auto gpuStart = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    const auto gpuEnd   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    std::cout << "GPU time: " << static_cast<double>(gpuEnd - gpuStart) / 1e6 << " ms\n";

    cv::Mat gpuResult(height, width, CV_8UC3);
    queue.enqueueReadBuffer(outputBuf, CL_TRUE, 0, size, gpuResult.data);
    cv::imwrite("output_gpu.png", gpuResult);

    // --- CPU ---
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    const cv::Mat cpuResult = blur_cpu(img);
    const auto cpuEnd = std::chrono::high_resolution_clock::now();

    const double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "CPU time: " << cpuMs << " ms\n";

    cv::imwrite("output_cpu.png", cpuResult);

    // --- CPU multi-threaded ---
    const auto cpuMtStart = std::chrono::high_resolution_clock::now();
    const cv::Mat cpuMtResult = blur_cpu_mt(img);
    const auto cpuMtEnd = std::chrono::high_resolution_clock::now();

    const double cpuMtMs = std::chrono::duration<double, std::milli>(cpuMtEnd - cpuMtStart).count();
    std::cout << "CPU MT time (" << std::thread::hardware_concurrency() << " threads): " << cpuMtMs << " ms\n";

    cv::imwrite("output_cpu_mt.png", cpuMtResult);

    return 0;
}
