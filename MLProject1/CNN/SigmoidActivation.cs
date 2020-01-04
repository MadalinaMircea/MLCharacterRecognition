using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class SigmoidActivation : Activation
    {
        public override LayerOutput Activate(LayerOutput output)
        {
            if(output is FlattenedImage)
            {
                return ActivateFlattenedImage((FlattenedImage)output);
            }
            if (output is FilteredImage)
            {
                return ActivateFilteredImage((FilteredImage)output);
            }

            return null;
        }

        private LayerOutput ActivateFlattenedImage(FlattenedImage output)
        {
            for(int i = 0; i < output.Size; i++)
            {
                output.Values[i] = ActivateValue(output.Values[i]);
            }

            return output;
        }

        private LayerOutput ActivateFilteredImage(FilteredImage img)
        {
            foreach (FilteredImageChannel channel in img.Channels)
            {
                for (int i = 0; i < channel.Size; i++)
                {
                    for (int j = 0; j < channel.Size; j++)
                    {
                        channel.Values[i, j] = ActivateValue(channel.Values[i, j]);
                    }
                }
            }

            return img;
        }


        private double ActivateValue(double v)
        {
            return 1.0 / (1.0 + Math.Exp(-v));
        }

        public override string ToString()
        {
            return "sigmoid";
        }

        public override LayerOutput GetDerivative(LayerOutput output)
        {
            if(output is FilteredImage)
            {
                return GetFilteredDerivative((FilteredImage)output);
            }
            if(output is FlattenedImage)
            {
                return GetFlattenedDerivative((FlattenedImage)output);
            }
            return null;
        }

        private LayerOutput GetFlattenedDerivative(FlattenedImage output)
        {
            for(int i = 0; i < output.Size; i++)
            {
                output.Values[i] = GetValueDerivative(output.Values[i]);
            }

            return output;
        }

        private LayerOutput GetFilteredDerivative(FilteredImage output)
        {
            for(int c = 0; c < output.NumberOfChannels; c++)
            {
                for(int i = 0; i <output.Size; i++)
                {
                    for(int j = 0; j < output.Size; j++)
                    {
                        output.Channels[c].Values[i, j] = GetValueDerivative(output.Channels[c].Values[i, j]);
                    }
                }
            }

            return output;
        }

        private double GetValueDerivative(double v)
        {
            return v * (1 - v);
        }
    }
}
