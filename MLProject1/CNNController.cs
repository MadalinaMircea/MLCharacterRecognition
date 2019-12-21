using MLProject1.CNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1
{
    class CNNController
    {
        public void CreateModel()
        {
            int imgWidth = 100;
            int imgHeight = 75;
            int outputSize = 26;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            ConvolutionalNeuralNetwork newModel = new ConvolutionalNeuralNetwork(imgWidth, imgHeight, outputSize);
            newModel.Add(new ConvolutionalLayer(32, 10, 10, reluActivation));
            newModel.Add(new MaxPoolingLayer());
            newModel.Add(new ConvolutionalLayer(64, 10, 10, reluActivation)); ;
            newModel.Add(new MaxPoolingLayer());
            newModel.Add(new DropoutLayer(0.2));
            newModel.Add(new ConvolutionalLayer(128, 5, 5, reluActivation));
            newModel.Add(new FlattenLayer());
            newModel.Add(new DenseLayer(512, reluActivation));
            newModel.Add(new DropoutLayer(0.5));
            newModel.Add(new DenseLayer(26, softmaxActivation));
        }
    }
}
