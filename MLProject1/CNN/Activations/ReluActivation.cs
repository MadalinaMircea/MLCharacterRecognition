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
                return value;
        }

        public override double GetValueDerivative(double v)
        {
            return (v <= 0) ? 0 : 1;
        }
    }
}
