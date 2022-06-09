#include "culia.h"

#include <device_launch_parameters.h>
#include <surface_functions.h>

#include <cstring>
#include <cstdint>
using namespace std;


__global__ void render_julia_set_kernel(cudaSurfaceObject_t surface, int width, int height,
    float_t c_real, float_t c_imag, float_t sqr_bailout, float_t pixel_size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float_t z_real = pixel_size * (float_t(x) - float_t(width) / 2.0f);
    float_t z_imag = pixel_size * (float_t(y) - float_t(height) / 2.0f);
    uint32_t i = 0;
    for (; i < 1024; ++i) {
        float_t z_real_new = z_real * z_real - z_imag * z_imag + c_real;
        float_t z_imag_new = 2 * z_real * z_imag + c_imag;
        if (z_real_new * z_real_new + z_imag_new * z_imag_new > sqr_bailout) break;
        z_real = z_real_new;
        z_imag = z_imag_new;
    }

    uint32_t data = 0xFF000000 + min(16 * i, 255) + 256*min(i, 255);
    surf2Dwrite(data, surface, 4 * x, y);
}

cudaError_t render_julia_set(cudaGraphicsResource_t cuda_renderbuffer, int width, int height, complex_t c)
{
    cudaError_t cuda_err;

    // map buffer for writing from CUDA
    cuda_err = cudaGraphicsMapResources(1, &cuda_renderbuffer);
    if (cuda_err != cudaSuccess) return cuda_err;

    cudaArray_t cuda_array;
    cuda_err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_renderbuffer, 0, 0);
    if (cuda_err != cudaSuccess) return cuda_err;

    // create surface object
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_array;
    cudaSurfaceObject_t cuda_surface;
    cudaCreateSurfaceObject(&cuda_surface, &resDesc);

    const float_t sqr_bailout = 16.0f;
    const float_t julia_width = 5.0f;
    const float_t pixel_size = julia_width / width;

    // execute kernel
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    render_julia_set_kernel<<<dim_grid, dim_block>>>(cuda_surface, width, height, c.real(), c.imag(), sqr_bailout, pixel_size);

    // destroy surface object
    cudaDestroySurfaceObject(cuda_surface);

    // unmap buffer
    cudaGraphicsUnmapResources(1, &cuda_renderbuffer);

    return cudaSuccess;
}