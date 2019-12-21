using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class DropoutLayer : NetworkLayer
    {
        public double Rate { get; set; }

        [JsonConstructor]
        public DropoutLayer(double rate) : base("Dropout")
        {
            Rate = rate;
        }
    }
}
