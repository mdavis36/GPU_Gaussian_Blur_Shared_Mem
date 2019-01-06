#include "bmp.h"



int endian4ByteToInt(char * buff)
{
      return int((unsigned char)(buff[3]) << 24 |
                 (unsigned char)(buff[2]) << 16 |
                 (unsigned char)(buff[1]) << 8 |
                 (unsigned char)(buff[0]));
}


int getIntFromBinaryStream(std::ifstream * f, int start_ind)
{
      char buff[4];
      f->seekg(start_ind, f->beg);
      f->read(&buff[0], 4);
      return endian4ByteToInt(&buff[0]);
}


void loadHeaderToByteArray(std::ifstream * f, unsigned char *out_buff)
{
      f->seekg(0, f->beg);
      f->read(reinterpret_cast<char *>(out_buff), 54);
}


void loadRGBDataFromFile(std::ifstream *f, BMP_IMAGE *bmp)
{
      long i;
      char padding[3];
      char c_r,c_g,c_b;

      f->seekg(54, f->beg);

      for (i = 0; i < bmp->size; i++)
      {
            bmp->red_buff[i] = -1; bmp->blue_buff[i] = -1; bmp->green_buff[i] = -1;

            f->read(&c_b, 1);
            f->read(&c_g, 1);
            f->read(&c_r, 1);

            if ((i+1) % bmp->width == 0) f->read(padding, 4 - ((bmp->width*3) % 4));

            bmp->red_buff[i]   = ((unsigned char)(c_r));
            bmp->blue_buff[i]  = ((unsigned char)(c_b));
            bmp->green_buff[i] = ((unsigned char)(c_g));
      }
}


void writeRGBDataToCharArray(BMP_IMAGE *bmp)
{
      int i, x, y, pad;
      int ind = 54;

      for (i = 0; i < ind; i++) bmp->write_data[i] = bmp->header_data[i];
      for (y = 0; y < bmp->height; y++)
      {
            for (x = 0; x < bmp->width; x++)
            {
                  bmp->write_data[ind]   = MIN(bmp->blue_buff [y * bmp->width + x], 255);
                  bmp->write_data[ind+1] = MIN(bmp->green_buff[y * bmp->width + x], 255);
                  bmp->write_data[ind+2] = MIN(bmp->red_buff  [y * bmp->width + x], 255);
                  ind += 3;
            }
            pad = 4 - ((bmp->width*3) % 4);
            ind += pad;
      }
}


void writeCharArrayToFile(char* f_name, BMP_IMAGE *bmp)
{
      std::ofstream* out_img;
      out_img = new std::ofstream;
      out_img->open(f_name, std::ifstream::binary);
      out_img->write(reinterpret_cast<char*>(bmp->write_data), bmp->file_size);
}


void loadBMPImage(char * name, BMP_IMAGE* bmp)
{
      bmp->f_name = name;

      /* Read in the bmp file */
      std::ifstream* bmp_file;
      bmp_file = new std::ifstream;
      bmp_file->open(bmp->f_name, std::ifstream::binary);

      /* Load heder data into char array to write back the file later */
      loadHeaderToByteArray(bmp_file, bmp->header_data);

      /* Inititalize the size of the write array */
      bmp->file_size = getIntFromBinaryStream(bmp_file, 2);
      bmp->write_data = new unsigned char[bmp->file_size];

      /* Retrieve the Width and Height of the Image */
      bmp->width  = getIntFromBinaryStream(bmp_file, 18);
      bmp->height = getIntFromBinaryStream(bmp_file, 22);

      /* Inititalize the RGB colour buffers */
      bmp->size = bmp->width * bmp->height;
      bmp->red_buff   = new unsigned char[bmp->size]; 
	bmp->green_buff = new unsigned char[bmp->size]; 
	bmp->blue_buff  = new unsigned char[bmp->size];

      /* Load RGB data from the image */
      loadRGBDataFromFile(bmp_file, bmp);

      std::cout << "File Size : "<< bmp->file_size << std::endl;
      std::cout << "Dimensions : " << bmp->width << " x " << bmp->height << std::endl;
}


void saveBMPImage(char *f_name, BMP_IMAGE *bmp)
{
      writeRGBDataToCharArray(bmp);
      writeCharArrayToFile(f_name, bmp);
}
