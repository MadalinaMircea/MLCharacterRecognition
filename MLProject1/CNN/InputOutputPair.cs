using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class InputOutputPair
    {
        public FilteredImage Input { get; set; }
        public double[] Output { get; set; }

        public InputOutputPair(FilteredImage input, double[] output)
        {
            Input = input;
            Output = output;
        }

        public InputOutputPair(string input, string output)
        {
            Input = ImageProcessing.GetNormalizedFilteredImage(new System.Drawing.Bitmap(input));
            Output = OneHotEncode(output);
        }

        private double[] OneHotEncode(string output)
        {
            double[] result = new double[26];
            result[output[output.Length - 1] - 65] = 1;

            return result;
        }
    }
}
