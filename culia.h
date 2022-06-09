#pragma once

#include "types.h"

#include <cuda_runtime.h>


cudaError_t render_julia_set(cudaGraphicsResource_t cuda_renderbuffer, int width, int height, complex_t c);
