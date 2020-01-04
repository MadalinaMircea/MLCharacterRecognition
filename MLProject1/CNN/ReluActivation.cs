using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class ReluActivation : PiecewiseActivation
    {
        public override string ToString()
        {
            return "relu";
        }

        public override double ActivateValue(double value)
        {
            if (value < 0)
                return 0;
            else
                //return (value > 1) ? 1 : value;
                return value;
        }
        

        //public override LayerOutput Activate(LayerOutput output)
        //{
        //    if(output is FlattenedImage)
        //    {
        //        return ActivateFlattenedImage((FlattenedImage)output);
        //    }
        //    else if(output is FilteredImage)
        //    {
        //        return ActivateFilteredImage((FilteredImage)output);
        //    }

        //    return null;
        //}

        //public override LayerOutput GetDerivative(LayerOutput output)
        //{
        //    if (output is FilteredImage)
        //    {
        //        return GetFilteredDerivative((FilteredImage)output);
        //    }
        //    if (output is FlattenedImage)
        //    {
        //        return GetFlattenedDerivative((FlattenedImage)output);
        //    }
        //    return null;
        //}

        

        public override double GetValueDerivative(double v)
        {
            return (v <= 0) ? 0 : 1;
        }
    }
}
