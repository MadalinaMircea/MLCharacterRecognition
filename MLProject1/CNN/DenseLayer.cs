using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class DenseLayer : NetworkLayer
    {
        public int Units { get; }
        public IActivation ActivationFunction { get; }
        public DenseLayer(int units, IActivation activationFunction) : base("Dense")
        {
            Units = units;
            ActivationFunction = activationFunction;
        }

        [JsonConstructor]
        public DenseLayer(int units, string activationFunction) : base("Dense")
        {
            Units = units;
            if (activationFunction == "relu")
            {
                ActivationFunction = new ReluActivation();
            }
            else if (activationFunction == "softmax")
            {
                ActivationFunction = new SoftmaxActivation();
            }
        }
    }
}
