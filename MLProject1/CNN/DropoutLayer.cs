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

        public FlattenedImage Output { get; set; }

        [JsonConstructor]
        public DropoutLayer(double rate) : base("Dropout")
        {
            Rate = rate;
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void ComputeOutput()
        {
            throw new NotImplementedException();
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            if (Output == null)
            {
                Output = new FlattenedImage(previousLayer.GetData().NumberOfWeights);
            }
        }
    }
}
