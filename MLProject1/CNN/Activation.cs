using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    abstract class Activation
    {
        public abstract FlattenedImage Activate(FlattenedImage img);

        public virtual FilteredImage Activate(FilteredImage img) { throw new Exception(); }

        public virtual FlattenedImage GetDerivative(FlattenedImage output) { throw new Exception(); }

        public virtual FlattenedImage GetDerivative(FlattenedImage output, int correctClass) { throw new Exception(); }

        public virtual FilteredImage GetDerivative(FilteredImage output) { throw new Exception(); }
    }
}
