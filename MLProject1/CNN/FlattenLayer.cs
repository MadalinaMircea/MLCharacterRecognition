using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class FlattenLayer : NetworkLayer
    {
        public FlattenedImage Output { get; set; }
        public FlattenLayer(FlattenedImage output = null) : base("Flatten")
        {
            Output = output;
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
            if(Output == null)
            {
                Output = new FlattenedImage(previousLayer.GetData().NumberOfWeights);
            }
        }
    }
}
