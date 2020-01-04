using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class EvaluationMetrics
    {
        public double Error { get; set; }
        public double Accuracy { get; set; }

        public EvaluationMetrics(double error, double accuracy)
        {
            Error = error;
            Accuracy = accuracy;
        }
    }
}
