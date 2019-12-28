using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    abstract class Activation
    {
        public abstract LayerOutput Activate(LayerOutput output);
    }
}
