using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class Unit
    {
        public double[] Weights { get; set; }
        public int NumberOfWeights { get; set; }

        [JsonIgnore]
        public double Output { get; set; }

        public Unit(int numberOfWeights)
        {
            NumberOfWeights = numberOfWeights;
            InitializeRandom();
        }

        [JsonConstructor]
        public Unit(int numberOfWeights, double[] weights)
        {
            NumberOfWeights = numberOfWeights;
            Weights = weights;
        }

        private void InitializeRandom()
        {
            Random rnd = new Random();

            Weights = new double[NumberOfWeights];

            for(int i = 0; i < NumberOfWeights; i++)
            {
                Weights[0] = rnd.NextDouble();
            }
        }

        public double ComputeOutput(FlattenedImage image)
        {
            double total = 0;

            for(int i = 0; i < NumberOfWeights; i++)
            {
                total += Weights[i] * image.Values[i];
            }

            Output = total;
            return total;
        }
    }
}
