#ifndef GAUS_H
#define GAUS_H

#include <iostream>
#include <math.h>


typedef struct FILTERS
{
      float *weights;
      int width;
} FILTER;


void generateFilter(FILTER *filter, int width, int sd_count)
{
      float a = 0.5 * (width - 1);
      float sig = (2/sd_count) * (a + 1);
      float _2sig2 = 2*sig*sig;

      float *gaus_vec = new float[width];

      for (int i = 0; i < a; i++){
            gaus_vec[i] = exp( (-1 * (i-a) * (i-a)) / _2sig2 );
            gaus_vec[width - i - 1] = gaus_vec[i];
      }
}


void bmpToFilter(BMP_IMAGE bmp, FILTER *filter)
{
      int i;
      filter->width = bmp.width;
      filter->weights = new float[filter->width * filter->width];
      for (i = 0; i < filter->width * filter->width; i ++)
      {
            filter->weights[i] = ((float)bmp.red_buff[i] + (float)bmp.green_buff[i] + (float)bmp.blue_buff[i] ) / 3.0f;
      }
}


void normalizeFilter(FILTER* filter)
{
      int i;
      float sum = 0;
      for (i = 0; i < filter->width * filter->width; i++) sum += filter->weights[i];
      for (i = 0; i < filter->width * filter->width; i++) filter->weights[i] = filter->weights[i] / (sum * 1.0f);
}


bool isInBounds(const int width, const int height, const int x, const int y)
{
      return x < width && x >= 0 && y < height && y >=0 ? true : false;
}


int _2dto1d(int w, int x, int y)
{
      return((y * w) + x);
}


void serialGaussianBlur(FILTER filter, BMP_IMAGE *bmp)
{
      int x, y, f_x, f_y, i;
      int f_offset;
      float r_val, g_val, b_val;
      int x_sam, y_sam;
      int* r, *g, *b;

      if (filter.width % 2 != 1) return;
      f_offset = -((filter.width - 1) / 2);

      int w    = bmp->width;
      int h    = bmp->height;
      int size = bmp->size;

      r = new int[size];
      g = new int[size];
      b = new int[size];

      for (x = 0; x < w; x++)
      {
            for (y = 0; y < h; y++)
            {
                  r_val = 0.0f;
                  g_val = 0.0f;
                  b_val = 0.0f;

                  for (f_x = 0; f_x < filter.width; f_x++)
                  {
                        for (f_y = 0; f_y < filter.width; f_y++)
                        {

                              x_sam = x + f_x + f_offset;
                              y_sam = y + f_y + f_offset;

                              if (isInBounds(w, h, x_sam, y_sam) == false)
                              {
                                    x_sam = x;
                                    y_sam = y;
                              }

                              r_val += ((float) bmp->red_buff  [_2dto1d(w, x_sam, y_sam)] * (float) filter.weights[_2dto1d(filter.width, f_x, f_y)]);
                              g_val += ((float) bmp->green_buff[_2dto1d(w, x_sam, y_sam)] * (float) filter.weights[_2dto1d(filter.width, f_x, f_y)]);
                              b_val += ((float) bmp->blue_buff [_2dto1d(w, x_sam, y_sam)] * (float) filter.weights[_2dto1d(filter.width, f_x, f_y)]);
                        }
                  }

                  r[_2dto1d(w, x, y)] = (int)(r_val);
                  g[_2dto1d(w, x, y)] = (int)(g_val);
                  b[_2dto1d(w, x, y)] = (int)(b_val);
            }
      }

      for (i = 0; i < w * h; i++)
      {
                  bmp->red_buff[i]   = r[i];
                  bmp->green_buff[i] = g[i];
                  bmp->blue_buff[i]  = b[i];
      }
}


#endif
