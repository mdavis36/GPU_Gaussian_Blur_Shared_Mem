#ifndef BMP_H
#define BMP_H

#include <fstream>
#include <iostream>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


typedef struct BMP_IMAGES
{
      char *f_name;
      int width;
      int height;
      int size;
      int file_size;
      unsigned char header_data[54];
      unsigned char *write_data;
      unsigned char *red_buff;
      unsigned char *green_buff;
      unsigned char *blue_buff;
} BMP_IMAGE;


int endian4ByteToInt(char * buff);

int getIntFromBinaryStream(std::ifstream * f, int start_ind);

void loadHeaderToByteArray(std::ifstream * f, unsigned char *out_buff);

void loadRGBDataFromFile(std::ifstream *f, BMP_IMAGE *bmp);

void writeRGBDataToCharArray(BMP_IMAGE *bmp);

void writeCharArrayToFile(char* f_name, BMP_IMAGE *bmp);

void loadBMPImage(char * name, BMP_IMAGE* bmp);

void saveBMPImage(char *f_name, BMP_IMAGE *bmp);

#endif
