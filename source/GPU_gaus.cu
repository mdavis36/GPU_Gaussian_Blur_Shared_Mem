// -----------------------------------------------------------------------------
// Filename : GPU_gaus.cu
// Created  : 04/10/2018
// Author   : Michael Davis
//
// Description : .cu CUDA C code designed to perform GPU accelerated Gaussian Blur
//               on 24bit bitmap RGB data.
//
// -----------------------------------------------------------------------------
#include "GPU_gaus.h"
#include <iostream>


// -----------------------------------------------------------------------------
// Function Name : gpuErrchk, gpuAssert
//
// Description : CUDA call error checking and handling. This C function was copied
//               from stack overflows website from user talonmies comment.
//
// Credit :
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// -----------------------------------------------------------------------------
extern "C" {
      #include <stdio.h>
      #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
      inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
      {
         if (code != cudaSuccess)
         {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
         }
      }
}



// -----------------------------------------------------------------------------
// Function Name : cuda2dto1d
//
// Description : Device function to be run and called by the GPU.
//               Helper function to convert 2D coordinates to a 1D index.
//
// Inputs : w - Width of 2D co-ordinate space.
//          x, y - Co-ords to be converted.
// -----------------------------------------------------------------------------
__device__ int cuda2dto1d(int w, int x, int y)
{
      return ((y * w) + x);
}



// -----------------------------------------------------------------------------
// Function Name : cudaIsInBounds
//
// Description : Device function to be run and called by the GPU.
//               Helper function to check if a 2D co-ordinate position is in bounds.
//
// Inputs : width, height - Width and Height of 2D co-ordinate space.
//          x, y - Co-ords to be checked.
// -----------------------------------------------------------------------------
__device__ bool cudaIsInBounds(int width, int height, int x, int y)
{
      return x < width && x >= 0 && y < height && y >=0 ? true : false;
}


// -----------------------------------------------------------------------------
// Function Name : getSharedGlobalCoOrds
//
// Description : retrieve cartesian coordinate values of a 2D shared memory structure.
//
// Inputs : sindx - relative index in shared memory.
//		bpos  - gloabal 2D coordinates of 2D cuda block.
//		sDim  - Dimensions of 2D shared memory.
//		w, h  - width and height of image.
//		hw	- half width of filter (filterwidth - 1) / 2
//
// Output : out - out co-ordinates relative to the image of the shared memory
//		bool - if the coordinates are within the image or not.
//         
// -----------------------------------------------------------------------------

__device__ bool getSharedGlobalCoOrds(int sindx, dim3 bpos, dim3 sDim, int w, int h, int hw, dim3 *out)
{
	dim3 spos(sindx%sDim.x, sindx/sDim.x);
	dim3 gpos(bpos.x+spos.x-hw, bpos.y+spos.y-hw);
	if (cudaIsInBounds(w, h, gpos.x, gpos.y))
	{
		*out = gpos;
		return true;
	}
	return false;
}


// -----------------------------------------------------------------------------
// Function Name : getSharedIndx
//
// Description : retrieve realtive shared memory index based on 2D global coordinates.
//
// Inputs : gpos  - globaal 2D coordinates of 2D position.
//		bpos  - gloabal 2D coordinates of 2D cuda block.
//		sDim  - Dimensions of 2D shared memory.
//		hw	- half width of filter (filterwidth - 1) / 2
//
// Output : indx - out index relative to the image of the shared memory
//		bool - if the coordinates are within the shared memory dimensions.
//         
// -----------------------------------------------------------------------------
__device__ bool getSharedIndx(dim3 gpos, dim3 bpos, dim3 sDim, int hw, int *indx)  
{
	dim3 spos(gpos.x - bpos.x + hw, gpos.y - bpos.y + hw);
	if (cudaIsInBounds(sDim.x, sDim.y, spos.x, spos.y))
	{
		*indx = spos.y * sDim.x + spos.x;
		return true;
	}
	return false;
}


// -----------------------------------------------------------------------------
// Function Name : cudaGaussianBlur
//
// Description : Global CUDA Kernel function that can be called from host code.
//               Uses CUDA acceleration to create gaussian blurred data.
//               Takes in original RGB channel data and uses a convolutional
//               filter of width fil_w to generate weighted blur values for each pixel.
//               Filter must be an odd valued width.
//
// Input : *f_buff, fil_w - Filter buffer data, should be normalized in order to maintain
//                          image "energy", this stops darkening or lightening effects of
//                          the image upon computation.
//
//         *r_buff, *g_buff, *b_buff - RGB image data channels inputs.
//
//         w, h - Width and heght of the image
//
//         *r_out, *g_out, *b_out - RGB image data channels output arrays.
// -----------------------------------------------------------------------------
__global__ void cudaGaussianBlur(float *f_buff,  /* In  - Filter buffer data         */
                                 int fil_w,      /* In  - Fileter width              */
                                 int fil_off,    /* In  - Filter offset value        */
                                 unsigned char *r,         /* In  - Red channel buffer data    */
                                 unsigned char *g,         /* In  - Green channel buffer data  */
                                 unsigned char *b,         /* In  - Blue channel buffer data   */
                                 int w,          /* In  - Image width                */
                                 int h,          /* In  - Image height               */
                                 unsigned char *r_out,     /* Out - Red channel buffer data    */
                                 unsigned char *g_out,     /* Out - Red channel buffer data    */
                                 unsigned char *b_out,      /* Out - Red channel buffer data    */
					   dim3 sharedDim
                                 )
{
      // Iterable variables
      int x, y, f_x, f_y;

      // Temporary float values for each colour channel
      float r_val, g_val, b_val;

      // Sample coordinates and 1D texture indexes
      int x_sam, y_sam;
      int fil_index, shared_index;

      x = blockIdx.x * blockDim.x + threadIdx.x;
      y = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = y * w + x;
	int local_index = threadIdx.y * blockDim.x + threadIdx.x;
	int localArea = blockDim.x * blockDim.y;

	int sharedArea = sharedDim.x * sharedDim.y;
	dim3 bpos(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);

	extern __shared__ unsigned char shared_mem[];
	unsigned char *s_r = shared_mem;
	unsigned char *s_g = (unsigned char*)&shared_mem[sharedArea];
	unsigned char *s_b = (unsigned char*)&s_g[sharedArea];

	
	int i = 0;
	while ((i * localArea) + (local_index) < sharedArea)
	{
		int sindx = (i * localArea) + local_index;
		dim3 gPos;
		if (getSharedGlobalCoOrds(sindx, bpos, sharedDim, w, h, -fil_off, &gPos))
		{
			s_r[sindx] = r[gPos.y * w + gPos.x];
			s_g[sindx] = g[gPos.y * w + gPos.x];
			s_b[sindx] = b[gPos.y * w + gPos.x];
		}
		i++;
	}

	__syncthreads();

      // Check index is wthin image bounds
      if (cudaIsInBounds(w, h, x, y))
      {
            // Initialize temporary values for colour channels
            r_val = 0.0f;
            g_val = 0.0f;
            b_val = 0.0f;

            // Begin filter convolution.
            for (f_x = 0; f_x < fil_w; f_x++)
            {
                  for (f_y = 0; f_y < fil_w; f_y++)
                  {

                        // Calculate sample coordinates for original image
                        x_sam = x + f_x + fil_off;
                        y_sam = y + f_y + fil_off;

                        // If ample co-ords out of bounds, sample from current pixel.
                        if (cudaIsInBounds(w, h, x_sam, y_sam) == false)
                        {
                              x_sam = x;
                              y_sam = y;
                        }

                        // Caluclate 1D array index values.
                        fil_index = cuda2dto1d(fil_w, f_x, f_y);
				
				if (getSharedIndx(dim3(x_sam, y_sam), bpos, sharedDim, -fil_off, &shared_index))
				{ 

					// Increment temporary pixel colour values.
					r_val += ((float) s_r[shared_index] * (float) f_buff[fil_index]);
					g_val += ((float) s_g[shared_index] * (float) f_buff[fil_index]);
					b_val += ((float) s_b[shared_index] * (float) f_buff[fil_index]);
				}
                  }
            }

            // Assign values to output arrays.
          	r_out[ind] = (unsigned char)(r_val);
          	g_out[ind] = (unsigned char)(g_val);
          	b_out[ind] = (unsigned char)(b_val);
      }
}



// -----------------------------------------------------------------------------
// Function Name : GPUAcceleratedGaussianBlur
//
// Description : Callable GPU acceslerated gaussian blur.
//               Uses CUDA acceleration to create gaussian blurred data.
//               Takes in original RGB channel data and uses a convolutional
//               filter of width fil_w to generate weighted blur values for each
//               pixel. Filter must be an odd valued width.
//
// Input : *f_buff, fil_w - Filter buffer data, should be normalized in order to
//                          maintain image "energy", this stops darkening or
//                          lightening effects of the image upon computation.
//
//         *r_buff, *g_buff, *b_buff - RGB image data channels.
//
//         w, h - Width and heght of the image
// -----------------------------------------------------------------------------
void GPUAcceleratedGaussianBlur(float *f_buff, /* In  - Filter buffer data  */
                                int fil_w,     /* In  - Fileter width       */
                                unsigned char *r_buff,   /* Out - Red channel data    */
                                unsigned char *g_buff,   /* Out - Green channel data  */
                                unsigned char *b_buff,   /* Out - Blue channel data   */
                                int w,         /* In  - Image width         */
                                int h)         /* In  - Image height        */
{

      // ---- Check filter size and generate offset value
      if (fil_w % 2 != 1) return;
      int f_offset = -((fil_w - 1) / 2);

      // ---- Define size variables to allocate space on the GPU ----
      size_t bmp_size = w * h * sizeof(unsigned char);
      size_t fil_size = fil_w * fil_w * sizeof(float);

      // ---- Allocate device memory ----
      unsigned char *d_r, *d_g, *d_b;
      unsigned char *d_r_out, *d_g_out, *d_b_out;
      float *d_f;
      gpuErrchk( cudaMalloc(&d_f, fil_size) );

      gpuErrchk( cudaMalloc(&d_r, bmp_size) );
      gpuErrchk( cudaMalloc(&d_g, bmp_size) );
      gpuErrchk( cudaMalloc(&d_b, bmp_size) );

      gpuErrchk( cudaMalloc(&d_r_out, bmp_size) );
      gpuErrchk( cudaMalloc(&d_g_out, bmp_size) );
      gpuErrchk( cudaMalloc(&d_b_out, bmp_size) );

      // ---- Copy data from host memory to device memory ----
      gpuErrchk( cudaMemcpy(d_r, r_buff,  bmp_size, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_g, g_buff,  bmp_size, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_b, b_buff,  bmp_size, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_f, f_buff,  fil_size, cudaMemcpyHostToDevice) );

      // ---- Determine kernel launch dimensions --- -
	dim3 bDim = dim3(16, 16);
	dim3 gDim = dim3( 1 + (w + bDim.x - 1) / bDim.x, 1 + (h + bDim.y - 1) / bDim.y );
	int shared_size = (bDim.x + fil_w - 1) * (bDim.y + fil_w - 1) * 3 * sizeof(unsigned char);
	std::cout << "shared size : " << shared_size << std::endl;
	dim3 sharedDim = dim3(bDim.x + fil_w - 1, bDim.y + fil_w - 1);

      // --- Launch Kernel ----
      cudaGaussianBlur<<<gDim, bDim, shared_size>>>(d_f, fil_w, f_offset, d_r, d_g, d_b, w, h, d_r_out, d_g_out, d_b_out, sharedDim);

      // ---- Read data back from device to host ----
      gpuErrchk( cudaMemcpy(r_buff, d_r_out, bmp_size, cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(g_buff, d_g_out, bmp_size, cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(b_buff, d_b_out, bmp_size, cudaMemcpyDeviceToHost) );


      // ---- Free up allocated memory ----
      gpuErrchk( cudaFree(d_f) );

      gpuErrchk( cudaFree(d_r) );
      gpuErrchk( cudaFree(d_g) );
      gpuErrchk( cudaFree(d_b) );

      gpuErrchk( cudaFree(d_r_out) );
      gpuErrchk( cudaFree(d_g_out) );
      gpuErrchk( cudaFree(d_b_out) );

}
