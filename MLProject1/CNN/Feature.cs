using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class Feature : Unit
    {
        public RGBPixel[,] Filter;
        int Size;
        int Channels;

        public Feature(int size, int channels)
        {
            Filter = new RGBPixel[size, size];
            Size = size;
            Channels = channels;
        }

        public RGBPixel[,] Convolve(RGBPixel[,] image)
        {
            RGBPixel[,] result = new RGBPixel[Size, Size];

            return result;
        }
    }
}
