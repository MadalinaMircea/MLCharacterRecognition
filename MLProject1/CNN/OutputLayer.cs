using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class OutputLayer : NetworkLayer
    {
        public int Size { get; set; }

        public FlattenedImage Output { get; set; }

        public OutputLayer(int size) : base("Output")
        {
            Size = size;
        }

        public override void ComputeOutput()
        {
            throw new NotImplementedException();
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            Output = new FlattenedImage(Size);
        }
    }
}
