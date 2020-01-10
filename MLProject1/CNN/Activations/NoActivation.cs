using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class NoActivation : PiecewiseActivation
    {
        public override double ActivateValue(double value)
        {
            return 0 + value;
        }

        public override double GetValueDerivative(double value)
        {
            return 0 + value;
        }

        public override string ToString()
        {
            return "no";
        }
    }
}
