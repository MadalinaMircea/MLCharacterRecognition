﻿using Newtonsoft.Json;
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

        public string Padding { get; set; }

        public ConvolutionalLayer(int filterNumber, int filterSize, Activation activationFunction, string padding) : base("Convolutional")
        {
            FilterNumber = filterNumber;
            FilterSize = filterSize;
            ActivationFunction = activationFunction;

            Filters = new Filter[filterNumber];

            Padding = padding;
        }

        [JsonConstructor]
        public ConvolutionalLayer(int filterNumber, int filterSize, string activationFunction, string padding) : base("Convolutional")
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

            Padding = padding;
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

            //Task[] tasks = new Task[FilterNumber];

            //for (int i = 0; i < FilterNumber; i++)
            //{
            //    int taski = 0 + i;

            //    tasks[taski] = Task.Run(() =>
            //    {
            //        channels[taski] = Filters[taski].Convolve(img);
            //    });
            //}

            //Task.WaitAll(tasks);

            bool samePadding = (Padding == "same") ? true : false;

            for (int i = 0; i < FilterNumber; i++)
            {
                channels[i] = Filters[i].Convolve(img, samePadding);
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

            if(Padding == "same")
            {
                OutputImage = new FilteredImage(FilterNumber, previous.Size);
            }
            else
            {
                OutputImage = new FilteredImage(FilterNumber, previous.Size - FilterSize + 1);
            }
            
        }

        public override LayerOutput[] Backpropagate(LayerOutput[] nextOutput, double learningRate)
        {
            FilteredImage[] newErrors = new FilteredImage[FilterNumber];

            FilteredImage previous = (FilteredImage)PreviousLayer.GetData();

            FilteredImage nextErrors = (FilteredImage)nextOutput[0];
            FilteredImage activationDerivatives = (FilteredImage)ActivationFunction.GetDerivative(nextErrors);

            //Task[] tasks = new Task[FilterNumber];

            //for (int i = 0; i < FilterNumber; i++)
            //{
            //    int taski = 0 + i;

            //    tasks[taski] = Task.Run(() =>
            //    {
            //        newErrors[taski] = Filters[taski].Backpropagate(previous,
            //        activationDerivatives.Channels[taski], learningRate);
            //    });
            //}

            //Task.WaitAll(tasks);

            bool samePadding = (Padding == "same") ? true : false;

            for (int i = 0; i < FilterNumber; i++)
            {
                newErrors[i] = Filters[i].Backpropagate(previous,
                    activationDerivatives.Channels[i], learningRate, samePadding);
            }

            return newErrors;
        }
    }
}
