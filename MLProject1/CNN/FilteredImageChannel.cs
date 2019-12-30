using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class FilteredImageChannel
    {
        public double[,] Values { get; set; }
        public int Size { get; set; }

        public FilteredImageChannel(int size, double[,] values)
        {
            Size = size;
            Values = values;
        }

        public FilteredImageChannel(int size)
        {
            Size = size;
            Values = new double[size, size];
        }
    }
}
