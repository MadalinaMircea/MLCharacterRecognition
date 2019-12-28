using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class SoftmaxActivation : Activation
    {
        public override string ToString()
        {
            return "softmax";
        }

        public override LayerOutput Activate(LayerOutput output)
        {
            if (output is FilteredImage)
            {
                return null;
            }

            FlattenedImage img = (FlattenedImage)output;

            double sum = 0;

            for(int i = 0; i < img.Size; i++)
            {
                img.Values[i] = Math.Exp(img.Values[i]);
                sum += img.Values[i];
            }

            for(int i = 0; i < img.Size; i++)
            {
                img.Values[i] /= sum;
            }

            return img;
        }
    }
}
