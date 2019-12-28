using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class Filter : WeightType
    {
        //public RGBPixel[,] Filter;
        public Kernel[] Kernels { get; set; }
        public int KernelNumber { get; set; }
        public int KernelSize { get; set; }

        public Filter(int kernelNumber, int kernelSize)
        {
            KernelNumber = kernelNumber;
            KernelSize = kernelSize;
            Kernels = new Kernel[KernelNumber];

            InitializeRandom();
        }

        public Filter(int kernelNumber, int kernelSize, Kernel[] kernels)
        {
            KernelNumber = kernelNumber;
            KernelSize = kernelSize;
            Kernels = kernels;
        }

        private void InitializeRandom()
        {
            for (int i = 0; i < KernelNumber; i++)
            {
                Kernels[i] = new Kernel(KernelSize);
            }
        }

        //public RGBPixel[,] Convolve(RGBPixel[,] image)
        //{
        //    RGBPixel[,] result = new RGBPixel[image.GetLength(0), image.GetLength(1)];

        //    int halfSize = KernelSize / 2;

        //    for(int imageI = halfSize; imageI < image.GetLength(0) - halfSize - 1; imageI++)
        //    {
        //        for(int imageJ = halfSize; imageJ < image.GetLength(1) - halfSize - 1; imageJ++)
        //        {
        //            //Working on pixel image[imageI, imageJ]

        //            double totalRed = 0, totalGreen = 0, totalBlue = 0;

        //            for(int filterI = 0; filterI < KernelSize; filterI++)
        //            {
        //                for(int filterJ = 0; filterJ < KernelSize; filterJ++)
        //                {
        //                    totalRed += image[imageI + filterI, imageJ + filterJ].Red * K[filterI, filterJ];
        //                    totalGreen += image[imageI + filterI, imageJ + filterJ].Green * Filter[filterI, filterJ];
        //                    totalBlue += image[imageI + filterI, imageJ + filterJ].Blue * Filter[filterI, filterJ];
        //                }
        //            }

        //            result[imageI - halfSize, imageJ - halfSize] = new RGBPixel(totalRed, totalGreen, totalBlue);
        //        }
        //    }

        //    return result;
        //}

        public FilteredImageChannel Convolve(FilteredImage input)
        {
            double[,] values = new double[KernelSize, KernelSize];

            double[,] kernelOutput;

            for (int i = 0; i < input.NumberOfChannels; i++)
            {
                kernelOutput = Kernels[i].Convolve(input.Channels[i]);

                for (int kernelI = 0; kernelI < KernelSize; kernelI++)
                {
                    for (int kernelJ = 0; kernelJ < KernelSize; kernelJ++)
                    {
                        values[kernelI, kernelJ] += kernelOutput[kernelI, kernelJ];
                    }
                }
            }

            return new FilteredImageChannel(KernelSize, values);
        }
    }
}
