using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class ConvolutionalLayer : NetworkLayer
    {
        public int FilterNumber { get; }
        public Filter[] Filters { get; set; }
        public int FilterSize { get; }
        public Activation ActivationFunction { get; }

        public FilteredImage OutputImage { get; set; }

        public ConvolutionalLayer(int filterNumber, int filterSize, Activation activationFunction, Filter[] filters) : base("Convolutional")
        {
            FilterNumber = filterNumber;
            Filters = filters;
            FilterSize = filterSize;
            ActivationFunction = activationFunction;
            OutputImage = new FilteredImage(filterNumber, new FilteredImageChannel[filterNumber]);
        }

        [JsonConstructor]
        public ConvolutionalLayer(int filterNumber, int filterSize, string activationFunction, Filter[] filters) : base("Convolutional")
        {
            FilterNumber = filterNumber;
            Filters = filters;
            FilterSize = filterSize;

            if (activationFunction == "relu")
            {
                ActivationFunction = new ReluActivation();
            }
            else if (activationFunction == "softmax")
            {
                ActivationFunction = new SoftmaxActivation();
            }

            OutputImage = new FilteredImage(filterNumber, new FilteredImageChannel[filterNumber]);
        }

        public ConvolutionalLayer(int filterNumber, int filterSize, Activation activationFunction) : base("Convolutional")
        {
            FilterNumber = filterNumber;
            FilterSize = filterSize;
            ActivationFunction = activationFunction;

            Filters = new Filter[filterNumber];

            OutputImage = new FilteredImage(filterNumber, new FilteredImageChannel[filterNumber]);
        }

        private void CreateKernels(int kernelNumber)
        {
            for (int i = 0; i < FilterNumber; i++)
            {
                Filters[i] = new Filter(kernelNumber, FilterSize);
            }
        }

        public override LayerOutput GetData()
        {
            return OutputImage;
        }

        public override void ComputeOutput()
        {
            FilteredImageChannel[] channels = new FilteredImageChannel[FilterNumber];
            FilteredImage img = (FilteredImage)PreviousLayer.GetData();

            for (int i = 0; i < FilterNumber; i++)
            {
                channels[i] = Filters[i].Convolve((FilteredImage)PreviousLayer.GetData());
            }
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            CreateKernels(previousLayer.GetData().NumberOfWeights);
        }
    }
}
