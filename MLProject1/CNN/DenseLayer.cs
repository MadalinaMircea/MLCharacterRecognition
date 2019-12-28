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
        public Activation ActivationFunction { get; }

        public FlattenedImage Output { get; set; }
        public DenseLayer(int units, Activation activationFunction, FlattenedImage output = null) : base("Dense")
        {
            Units = units;
            ActivationFunction = activationFunction;
            Output = output;
        }

        [JsonConstructor]
        public DenseLayer(int units, string activationFunction, FlattenedImage output = null) : base("Dense")
        {
            Units = units;
            Output = output;
            if (activationFunction == "relu")
            {
                ActivationFunction = new ReluActivation();
            }
            else if (activationFunction == "softmax")
            {
                ActivationFunction = new SoftmaxActivation();
            }
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void ComputeOutput()
        {
            throw new NotImplementedException();
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            if (Output == null)
            {
                Output = new FlattenedImage(Units);
            }
        }
    }
}
