using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class RGBPixel
    {
        public int Red { get; set; }
        public int Green { get; set; }
        public int Blue { get; set; }
        public RGBPixel(int red = 0, int green = 0, int blue = 0)
        {
            Red = red;
            Green = green;
            Blue = blue;
        }
    }
}
