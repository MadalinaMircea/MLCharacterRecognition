using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class SoftmaxActivation : IActivation
    {
        public override string ToString()
        {
            return "softmax";
        }
    }
}
