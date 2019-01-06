#include <iostream>
#include <stdio.h>
#include <math.h>

#include "bmp.h"
#include "gaus.h"
#include "GPU_gaus.h"
#include "timer.h"

#define GPU_ACCELERATION 1

int main()
{
      double t0, t1, t2, t3;

      BMP_IMAGE bmp_filter;
      FILTER my_filter;

      BMP_IMAGE my_bmp;

      t0 = getProcessTime();

      loadBMPImage((char *)"res/49_filter.bmp", &bmp_filter);
      bmpToFilter(bmp_filter, &my_filter);
      normalizeFilter(&my_filter);

      loadBMPImage((char *)"res/dog.bmp",&my_bmp);
      t1 = getProcessTime();

      if (GPU_ACCELERATION != 1)
      {
            serialGaussianBlur(my_filter, &my_bmp);
      } else {
            GPUAcceleratedGaussianBlur(my_filter.weights,
                                       my_filter.width,
                                       my_bmp.red_buff,
                                       my_bmp.green_buff,
                                       my_bmp.blue_buff,
                                       my_bmp.width,
                                       my_bmp.height);
      }

      t2 = getProcessTime();

      saveBMPImage((char *)"res/out.bmp", &my_bmp);
      t3 = getProcessTime();

      std::cout << "Initial data input time : " << t1 - t0 << " seconds" << std::endl;
      std::cout << "Image processing time   : " << t2 - t1 << " seconds" << std::endl;
      std::cout << "Data saving/output time : " << t3 - t2 << " seconds" << std::endl;

      return 0;
}
