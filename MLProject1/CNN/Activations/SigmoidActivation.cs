using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    public class SigmoidActivation : PiecewiseActivation
    {
        public override double ActivateValue(double v)
        {
            return 1.0 / (1.0 + Math.Exp(-v));
        }

        public override string ToString()
        {
            return "sigmoid";
        }

        public override double GetValueDerivative(double v)
        {
            return v * (1 - v);
        }
    }
}
