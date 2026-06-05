# Gaussian Blur — CPU vs GPU

Applies a 7×7 Gaussian blur to an image using three implementations and compares their performance.

## Implementations

| Implementation | Description |
|---|---|
| CPU single-threaded | Baseline — plain nested loops over every pixel |
| CPU multi-threaded | Rows split across `std::thread::hardware_concurrency()` threads |
| GPU (OpenCL) | Each pixel processed by a separate work-item; kernel dispatched via `enqueueNDRangeKernel` |

## Results

Test image: 960×1280 px, 3 channels — Intel UHD Graphics 620 / 8-thread CPU

| Implementation | Time |
|---|---|
| CPU single-threaded | 15851 ms |
| CPU multi-threaded (8 threads) | 2805 ms |
| GPU (OpenCL) | 10.07 ms |

The GPU is ~1000× faster than the single-threaded CPU version on this workload. This is because Gaussian blur is an embarrassingly parallel problem — every output pixel is independent — which maps directly onto the GPU's massively parallel architecture.

The OpenCL kernel is dispatched as a 2D NDRange (`width × height` work-items), one per pixel. Each work-item still loops over the 3 color channels sequentially — extending to a 3D NDRange would be valid, but unnecessary: with only 3 channels the GPU is already fully saturated by pixel-level parallelism, and the overhead of tripling the work-item count would outweigh any gain.

## Dependencies

- [OpenCV](https://opencv.org/) — image I/O (`cv::imread` / `cv::imwrite`)
- OpenCL — fetched automatically by CMake (Khronos headers + ICD loader + C++ bindings)

On Windows, install OpenCV via [vcpkg](https://vcpkg.io/):
```
vcpkg install opencv:x64-windows
```

## Build

```bash
cmake -B build
cmake --build build
```

CMake fetches the OpenCL headers, ICD loader, and C++ bindings automatically via `FetchContent`.

## Usage

Place your input image as `input.png` in the working directory, then run the executable. Three output files are written:

```
output_gpu.png       # GPU result
output_cpu.png       # CPU single-threaded result
output_cpu_mt.png    # CPU multi-threaded result
```

Timing for all three implementations is printed to stdout.
