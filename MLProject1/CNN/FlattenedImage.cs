using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class FlattenedImage : LayerOutput
    {
        public int Size { get; set; }
        public double[] Values { get; set; }
        public FlattenedImage(int size, double[] values = null)
        {
            Size = size;
            Values = values;
            NumberOfWeights = size;
        }
    }
}
