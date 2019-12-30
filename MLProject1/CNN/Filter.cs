using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class Filter
    {
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

        private void InitializeRandom()
        {
            for (int i = 0; i < KernelNumber; i++)
            {
                Kernels[i] = new Kernel(KernelSize);
            }
        }
        public FilteredImageChannel Convolve(FilteredImage input)
        {
            int resultSize = input.Size;
            double[,] values = new double[resultSize, resultSize];

            double[,] kernelOutput;

            for (int i = 0; i < input.NumberOfChannels; i++)
            {
                kernelOutput = Kernels[i].Convolve(input.Channels[i]);

                for (int outputI = 0; outputI < resultSize; outputI++)
                {
                    for (int outputJ = 0; outputJ < resultSize; outputJ++)
                    {
                        values[outputI, outputJ] += kernelOutput[outputI, outputJ];
                    }
                }
            }

            return new FilteredImageChannel(resultSize, values);
        }
    }
}
