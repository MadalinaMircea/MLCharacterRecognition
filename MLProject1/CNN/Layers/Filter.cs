using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public class Filter
    {
        public Kernel[] Kernels { get; set; }
        public int KernelNumber { get; set; }
        public int KernelSize { get; set; }

        [JsonConstructor]
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
        public FilteredImageChannel Convolve(FilteredImage input, bool samePadding)
        {
            int resultSize;

            if (samePadding)
            {
                resultSize = input.Size;
            }
            else
            {
                resultSize = input.Size - KernelSize + 1;
            }

            double[,] values = new double[resultSize, resultSize];

            double[,] kernelOutput;


            Task[] tasks = new Task[KernelNumber];

            for (int i = 0; i < input.NumberOfChannels; i++)
            {
                int taski = 0 + i;

                tasks[taski] = Task.Run(() =>
                {
                    kernelOutput = Kernels[taski].Convolve(input.Channels[taski], samePadding);

                    for (int outputI = 0; outputI < resultSize; outputI++)
                    {
                        for (int outputJ = 0; outputJ < resultSize; outputJ++)
                        {
                            Monitor.Enter(values);
                            values[outputI, outputJ] += kernelOutput[outputI, outputJ];
                            Monitor.Exit(values);
                        }
                    }
                });
            }

            Task.WaitAll(tasks);


            //for (int i = 0; i < input.NumberOfChannels; i++)
            //{
            //    kernelOutput = Kernels[i].Convolve(input.Channels[i], samePadding);

            //    for (int outputI = 0; outputI < resultSize; outputI++)
            //    {
            //        for (int outputJ = 0; outputJ < resultSize; outputJ++)
            //        {
            //            values[outputI, outputJ] += kernelOutput[outputI, outputJ];
            //        }
            //    }
            //}

            return new FilteredImageChannel(resultSize, values);
        }

        public FilteredImage Backpropagate(FilteredImage previous, FilteredImageChannel nextErrors, double learningRate, bool samePadding)
        {
            FilteredImageChannel[] newChannels = new FilteredImageChannel[KernelNumber];

            for(int k = 0; k < KernelNumber; k++)
            {
                newChannels[k] = Kernels[k].Backpropagate(previous.Channels[k], nextErrors, KernelNumber, learningRate, samePadding);
            }

            return new FilteredImage(KernelNumber, newChannels);
        }
    }
}
