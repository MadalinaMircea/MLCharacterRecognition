using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    [JsonConverter(typeof(NetworkLayerConverter))]
    abstract class NetworkLayer
    {
        public string Type { get; }

        [JsonConstructor]
        public NetworkLayer(string type)
        {
            Type = type;
        }
    }
}
