using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public class Unit
    {
        public double[] Weights { get; set; }
        public int NumberOfWeights { get; set; }

        public double Bias { get; set; }

        [JsonIgnore]
        public double Output { get; set; }

        public Unit(int numberOfWeights)
        {
            NumberOfWeights = numberOfWeights;
            InitializeRandom();
        }

        [JsonConstructor]
        public Unit(int numberOfWeights, double[] weights, double bias)
        {
            NumberOfWeights = numberOfWeights;
            Weights = weights;
            Bias = bias;
        }

        private void InitializeRandom()
        {
            Weights = new double[NumberOfWeights];

            for(int i = 0; i < NumberOfWeights; i++)
            {
                Weights[i] = GlobalRandom.GetRandomWeight() / NumberOfWeights;
            }

            Bias = 0;
        }

        public double ComputeOutput(FlattenedImage image)
        {
            double total = 0;

            for(int i = 0; i < NumberOfWeights; i++)
            {
                total += Weights[i] * image.Values[i];
            }

            Output = total + Bias;
            return Output;
        }
    }
}
