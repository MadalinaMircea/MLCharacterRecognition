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
        public int Filters { get; }
        public int WindowWidth { get; }
        public int WindowHeight { get; }

        public IActivation ActivationFunction { get; }
        public ConvolutionalLayer(int filters, int windowWidth, int windowHeight, IActivation activationFunction) : base("Convolutional")
        {
            Filters = filters;
            WindowWidth = windowWidth;
            WindowHeight = windowHeight;
            ActivationFunction = activationFunction;
        }

        [JsonConstructor]
        public ConvolutionalLayer(int filters, int windowWidth, int windowHeight, string activationFunction) : base("Convolutional")
        {
            Filters = filters;
            WindowWidth = windowWidth;
            WindowHeight = windowHeight;
            if (activationFunction == "relu")
            {
                ActivationFunction = new ReluActivation();
            }
            else if(activationFunction == "softmax")
            {
                ActivationFunction = new SoftmaxActivation();
            }
        }
    }
}
