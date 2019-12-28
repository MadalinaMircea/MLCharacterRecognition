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

        public FilteredImage Output { get; set; }

        [JsonConstructor]
        public MaxPoolingLayer(int pool = 2) : base("MaxPooling")
        {
            Pool = pool;
        }

        public override void ComputeOutput()
        {
            for(int i = 0; i < Output.NumberOfChannels; i++)
            {

            }
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            int weights = previousLayer.GetData().NumberOfWeights;
            Output = new FilteredImage(weights, new FilteredImageChannel[weights]);
        }
    }
}
