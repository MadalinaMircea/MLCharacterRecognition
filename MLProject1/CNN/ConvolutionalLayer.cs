using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class ConvolutionalLayer : NetworkLayer
    {
        public int FilterNumber { get; }

        [JsonIgnore]
        public Filter[] Filters { get; set; }
        public int FilterSize { get; }
        public Activation ActivationFunction { get; }

        [JsonIgnore]
        public FilteredImage OutputImage { get; set; }

        public ConvolutionalLayer(int filterNumber, int filterSize, Activation activationFunction) : base("Convolutional")
        {
            FilterNumber = filterNumber;
            FilterSize = filterSize;
            ActivationFunction = activationFunction;
        }

        [JsonConstructor]
        public ConvolutionalLayer(int filterNumber, int filterSize, string activationFunction) : base("Convolutional")
        {
            FilterNumber = filterNumber;
            FilterSize = filterSize;

            if (activationFunction == "relu")
            {
                ActivationFunction = new ReluActivation();
            }
            else if (activationFunction == "softmax")
            {
                ActivationFunction = new SoftmaxActivation();
            }

            Filters = new Filter[filterNumber];
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
                channels[i] = Filters[i].Convolve(img);
            }

            OutputImage = (FilteredImage)ActivationFunction.Activate(new FilteredImage(FilterNumber, channels));
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            FilteredImage previous = (FilteredImage)PreviousLayer.GetData();

            if (Filters[0] == null)
            {
                CreateKernels(previous.NumberOfChannels);
            }

            OutputImage = new FilteredImage(FilterNumber, previous.Size);
        }
    }
}
