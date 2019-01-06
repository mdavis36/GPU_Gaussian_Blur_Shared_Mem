#ifndef GPU_GAUS_H
#define GPU_GAUS_H

#include <iostream>

int runcudatest();


void GPUAcceleratedGaussianBlur(float *f_buff,
                                int fil_w,
                                unsigned char *r_buff,
                                unsigned char *g_buff,
                                unsigned char *b_buff,
                                int w,
                                int h);
#endif
