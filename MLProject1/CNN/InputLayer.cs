using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class InputLayer
    {
        public int Width { get; set; }
        public int Height { get; set; }

        public InputLayer(int width, int height)
        {
            Width = width;
            Height = height;
        }
    }
}
