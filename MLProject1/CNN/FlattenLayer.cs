using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class FlattenLayer : NetworkLayer
    {
        [JsonIgnore]
        public FlattenedImage Output { get; set; }
        public FlattenLayer(FlattenedImage output = null) : base("Flatten")
        {
            Output = output;
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void ComputeOutput()
        {
            FilteredImage image = (FilteredImage)PreviousLayer.GetData();
            int outputIndex = 0;

            for(int channel = 0; channel < image.NumberOfChannels; channel++)
            {
                for(int valuesI = 0; valuesI < image.Size; valuesI++)
                {
                    for(int valuesJ = 0; valuesJ < image.Size; valuesJ++)
                    {
                        Output.Values[outputIndex] = image.Channels[channel].Values[valuesI, valuesJ];
                        outputIndex++;
                    }
                }
            }
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            if(Output == null)
            {
                FilteredImage previous = (FilteredImage)previousLayer.GetData();
                Output = new FlattenedImage(previous.Size * previous.Size * previous.NumberOfChannels);
            }
        }
    }
}
