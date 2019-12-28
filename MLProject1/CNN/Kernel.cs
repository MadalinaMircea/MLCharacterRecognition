using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class Kernel
    {
        public double[,] Values { get; set; }
        public int Size { get; set; }

        public Kernel(int size, double[,] values)
        {
            if (size % 2 == 0)
                throw new Exception("Filter cannot have even size.");
            Size = size;
            Values = values;
        }

        public Kernel(int size)
        {
            if (size % 2 == 0)
                throw new Exception("Filter cannot have even size.");
            Size = size;
            InitializeRandom();
        }

        private void InitializeRandom()
        {
            Values = new double[Size, Size];
            
            Random rnd = new Random();

            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    Values[i, j] = rnd.NextDouble();
                }
            }
        }

        public double[,] Convolve(FilteredImageChannel input)
        {
            double[,] result = new double[input.Size, input.Size];
            int halfSize = Size / 2;

            for(int outputI = 0; outputI < input.Size; outputI++)
            {
                for(int outputJ = 0; outputJ < input.Size; outputJ++)
                {
                    double total = 0;
                    for(int kernelI = 0; kernelI < Size; kernelI++)
                    {
                        for(int kernelJ = 0; kernelJ < Size; kernelJ++)
                        {
                            int indexI = outputI + kernelI - halfSize;
                            int indexJ = outputJ + kernelJ - halfSize;

                            if(indexI >= 0 && indexI < Size && indexJ >= 0 && indexJ < Size)
                            {
                                total += input.Values[indexI, indexJ] * Values[kernelI, kernelJ];
                            }
                        }
                    }
                    result[outputI, outputJ] = total;
                }
            }
            return result;
        }
    }
}
