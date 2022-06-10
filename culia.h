#pragma once

#include "types.h"

#include <cuda_runtime.h>


/**
* Render julia set to CUDA registered render buffer.
*
* @param cuda_renderbuffer  render buffer to render to
* @param width              width of the buffer in pixels
* @param height             height of the buffer in pixels
* @param c                  Julia set c
*/
cudaError_t render_julia_set(cudaGraphicsResource_t cuda_renderbuffer, int width, int height, complex_t c);
