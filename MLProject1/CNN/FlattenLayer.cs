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
        public FlattenLayer() : base("Flatten")
        {

        }
    }
}
