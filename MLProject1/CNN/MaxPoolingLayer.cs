using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class MaxPoolingLayer : NetworkLayer
    {
        public int Pool { get; }

        [JsonIgnore]
        public FilteredImage Output { get; set; }

        [JsonConstructor]
        public MaxPoolingLayer(int pool = 2) : base("MaxPooling")
        {
            Pool = pool;
        }

        public override void ComputeOutput()
        {
            FilteredImage input = (FilteredImage)PreviousLayer.GetData();
            FilteredImageChannel[] outputChannels = new FilteredImageChannel[input.NumberOfChannels];
            int inputSize = input.Size;
            int outputSize = inputSize / Pool;

            for(int i = 0; i < input.NumberOfChannels; i++)
            {
                outputChannels[i] = new FilteredImageChannel(outputSize);
            }
            
            for (int channelI = 0; channelI + Pool < inputSize; channelI += Pool)
            {
                for(int channelJ = 0; channelJ + Pool < inputSize; channelJ += Pool)
                {
                    for(int poolI = 0; poolI < Pool; poolI++)
                    {
                        for(int poolJ = 0; poolJ < Pool; poolJ++)
                        {
                            for (int channel = 0; channel < input.NumberOfChannels; channel++)
                            {
                                FilteredImageChannel auxInput = input.Channels[channel];
                                FilteredImageChannel auxOutput = outputChannels[channel];

                                if (auxOutput.Values[channelI / Pool, channelJ / Pool] < auxInput.Values[channelI + poolI, channelJ + poolJ])
                                {
                                    auxOutput.Values[channelI / Pool, channelJ / Pool] = auxInput.Values[channelI + poolI, channelJ + poolJ];
                                }
                            }
                        }
                    }
                }
            }

            Output = new FilteredImage(input.NumberOfChannels, outputChannels);
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            FilteredImage previous = (FilteredImage)previousLayer.GetData();
            Output = new FilteredImage(previous.NumberOfChannels, previous.Size / Pool);
        }

        public override LayerOutput[] Backpropagate(LayerOutput[] nextOutput, double learningRate)
        {
            FilteredImage input = (FilteredImage)PreviousLayer.GetData();

            FilteredImageChannel[] outputChannels = new FilteredImageChannel[input.NumberOfChannels];

            FilteredImage nextErrors = (FilteredImage)nextOutput[0];

            int errorSize = nextErrors.Size;
            int outputSize = input.Size;

            for (int i = 0; i < input.NumberOfChannels; i++)
            {
                outputChannels[i] = new FilteredImageChannel(outputSize);
            }

            Task[] tasks = new Task[input.NumberOfChannels];

            for (int channel = 0; channel < input.NumberOfChannels; channel++)
            {
                int taskc = 0 + channel;

                tasks[taskc] = Task.Run(() =>
               {
                   for (int channelI = 0; channelI + Pool < outputSize; channelI += Pool)
                   {
                       for (int channelJ = 0; channelJ + Pool < outputSize; channelJ += Pool)
                       {
                           int maxi = -1, maxj = -1;
                           double maxx = -1;

                           for (int poolI = 0; poolI < Pool; poolI++)
                           {
                               for (int poolJ = 0; poolJ < Pool; poolJ++)
                               {
                                   int i = channelI + poolI, j = channelJ + poolJ;
                                   if (input.Channels[taskc].Values[i, j] > maxx)
                                   {
                                       maxx = input.Channels[taskc].Values[i, j];
                                       maxi = i;
                                       maxj = j;
                                   }
                               }
                           }

                           outputChannels[taskc].Values[maxi, maxj] =
                               nextErrors.Channels[taskc].Values[channelI / Pool, channelJ / Pool];
                       }
                   }
               });
            }

            Task.WaitAll(tasks);

            return new FilteredImage[1] { new FilteredImage(input.NumberOfChannels, outputChannels) };
        }
    }
}
