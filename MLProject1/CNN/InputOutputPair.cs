using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class InputOutputPair
    {
        //public FilteredImage Input { get; set; }

        public string Input { get; set; }
        public double[] Output { get; set; }

        public char OutputChar { get; set; }

        public InputOutputPair(string input, string output)
        {
            Input = input;
            OutputChar = output[output.Length - 1];
            Output = OneHotEncode(output);
        }

        private double[] OneHotEncode(string output)
        {
            double[] result = new double[26];
            result[OutputChar - 65] = 1;

            return result;
        }
    }
}
