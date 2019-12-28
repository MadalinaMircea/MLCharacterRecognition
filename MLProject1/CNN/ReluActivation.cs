using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class ReluActivation : Activation
    {
        public override string ToString()
        {
            return "relu";
        }

        private double ActivateValue(double value)
        {
            if (value < 0)
                return 0;
            else
                return value;
        }
        private LayerOutput ActivateFlattenedImage(FlattenedImage img)
        {
            for(int i = 0; i < img.Size; i++)
            {
                img.Values[i] = ActivateValue(img.Values[i]);
            }

            return img;
        }

        private LayerOutput ActivateFilteredImage(FilteredImage img)
        {
            foreach(FilteredImageChannel channel in img.Channels)
            {
                for(int i = 0; i < channel.Size; i++)
                {
                    for(int j = 0; j < channel.Size; j++)
                    {
                        channel.Values[i, j] = ActivateValue(channel.Values[i, j]);
                    }
                }
            }

            return img;
        }

        public override LayerOutput Activate(LayerOutput output)
        {
            if(output is FlattenedImage)
            {
                return ActivateFlattenedImage((FlattenedImage)output);
            }
            else if(output is FilteredImage)
            {
                return ActivateFilteredImage((FilteredImage)output);
            }

            return null;
        }
    }
}
