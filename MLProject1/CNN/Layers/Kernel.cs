using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public class Kernel
    {
        public double[,] Values { get; set; }
        public int Size { get; set; }

        [JsonIgnore]
        public double ElementSum { get; set; }

        [JsonConstructor]
        public Kernel(int size, double[,] values)
        {
            if (size % 2 == 0)
                throw new Exception("Filter cannot have even size.");
            Size = size;
            Values = values;

           // ComputeElementSum();
        }

        //private void ComputeElementSum()
        //{
        //    ElementSum = MatrixUtils.ElementSum(Values);
        //}

        public Kernel(int size)
        {
            if (size % 2 == 0)
                throw new Exception("Filter cannot have even size.");
            Size = size;
            InitializeRandom();
            //ComputeElementSum();
        }

        private void InitializeRandom()
        {
            Values = new double[Size, Size];

            int squaredSize = Size * Size;
            
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    Values[i, j] = GlobalRandom.GetRandomWeight() / squaredSize;
                }
            }
        }

        public double[,] Convolve(FilteredImageChannel input, bool samePadding)
        {
            //double[,] flippedKernel = MatrixUtils.Rotate180(Values);

            //int fullSize = Size * Size;

            if(samePadding)
            {
                return MatrixUtils.ConvolveSame(input.Values, Values);
            }
            
            return MatrixUtils.Convolve(input.Values, Values);

            //for (int i = 0; i < Size; i++)
            //{
            //    for (int j = 0; j < Size; j++)
            //    {
            //        convResult[i, j] /= fullSize;
            //    }
            //}

        }

        public FilteredImageChannel Backpropagate(FilteredImageChannel previous, FilteredImageChannel nextErrors, int totalKernels, double learningRate, bool samePadding)
        {
            //double[,] kernelDerivatives = new double[1,1];

            //Task t = Task.Run(() =>
            //{
            //    if (samePadding)
            //    {
            //        kernelDerivatives = MatrixUtils.ConvolveSame(previous.Values, nextErrors.Values);
            //    }
            //    else
            //    {
            //        kernelDerivatives = MatrixUtils.Convolve(previous.Values, nextErrors.Values);
            //    }
            //});

            //double[,] flippedKernel = MatrixUtils.Rotate180(Values);

            ////double[,] newErrors = MatrixUtils.Convolve(flippedKernel, nextErrors.Values);

            //double[,] newErrors;

            //if (samePadding)
            //{
            //    newErrors = MatrixUtils.FullConvolutionSame(flippedKernel, nextErrors.Values);
            //}
            //else
            //{
            //    newErrors = MatrixUtils.FullConvolution(flippedKernel, nextErrors.Values);
            //}



            //t.Wait();

            //for (int i = 0; i < Size; i++)
            //{
            //    for(int j = 0; j < Size; j++)
            //    {
            //        Values[i, j] -= learningRate * kernelDerivatives[i, j];
            //    }
            //}

            //ComputeElementSum();

            //return new FilteredImageChannel(Size, newErrors);

            //double[,] flippedErrors = MatrixUtils.Rotate180(nextErrors.Values);

            double[,] deltaWeights = new double[1, 1];

            Task t1 = Task.Run(() =>
            {
                if (samePadding)
                {
                    deltaWeights = MatrixUtils.ConvolveSame(previous.Values, nextErrors.Values);

                }
                else
                {
                    deltaWeights = MatrixUtils.Convolve(previous.Values, nextErrors.Values);
                }
            });
            

            //double[,] flippedKernel = MatrixUtils.Rotate180(Values);


            double[,] newErrors;

            if(samePadding)
            {
                newErrors = MatrixUtils.Convolve(Values, nextErrors.Values);
            }
            else
            {
                newErrors = MatrixUtils.FullConvolution(Values, nextErrors.Values);
            }

            t1.Wait();

            double[,] newWeights = new double[Size, Size];

            for(int i = 0; i < Size; i++)
            {
                for(int j = 0; j < Size; j++)
                {
                    newWeights[i, j] = Values[i, j] - learningRate * deltaWeights[i, j];
                }
            }

            //Values = MatrixUtils.Rotate180(newWeights);

            Values = newWeights;

            return new FilteredImageChannel(newErrors.GetLength(0), newErrors);

        }
    }
}
