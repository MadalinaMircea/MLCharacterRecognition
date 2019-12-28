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

        public NetworkLayer PreviousLayer { get; set; }

        [JsonConstructor]
        public NetworkLayer(string type)
        {
            Type = type;
        }

        public abstract void ComputeOutput();

        public abstract LayerOutput GetData();

        public abstract void CompileLayer(NetworkLayer previousLayer);
    }

    
}
