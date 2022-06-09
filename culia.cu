#include "culia.h"

#include "device_launch_parameters.h"
#include "surface_functions.h" 

#include <cstring>
#include <cstdint>

__global__ void test_kernel(cudaSurfaceObject_t surface, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uint32_t data = 0xFF0000FF;
        surf2Dwrite(data, surface, 4*x, y);
    }
}

cudaError_t render_julia_set(cudaGraphicsResource_t cuda_renderbuffer, int width, int height)
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

    // execute kernel
    dim3 dim_block(16, 16);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);
    test_kernel<<<dim_grid, dim_block>>>(cuda_surface, width, height);

    // destroy surface object
    cudaDestroySurfaceObject(cuda_surface);

    // unmap buffer
    cudaGraphicsUnmapResources(1, &cuda_renderbuffer);

    return cudaSuccess;
}