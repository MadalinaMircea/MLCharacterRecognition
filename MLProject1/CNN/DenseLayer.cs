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
        public int NumberOfUnits { get; }
        public Activation ActivationFunction { get; }

        [JsonIgnore]
        public Unit[] Units { get; set; }

        [JsonIgnore]
        public FlattenedImage Output { get; set; }
        public DenseLayer(int numberOfUnits, Activation activationFunction) : base("Dense")
        {
            NumberOfUnits = numberOfUnits;
            ActivationFunction = activationFunction;
        }

        [JsonConstructor]
        public DenseLayer(int numberOfUnits, string activationFunction) : base("Dense")
        {
            NumberOfUnits = numberOfUnits;
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
            FlattenedImage previous = (FlattenedImage)PreviousLayer.GetData();

            for (int i = 0; i < NumberOfUnits; i++)
            {
                Output.Values[i] = Units[i].ComputeOutput(previous);
            }

            Output = (FlattenedImage)ActivationFunction.Activate(Output);
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;
            FlattenedImage previous = (FlattenedImage)PreviousLayer.GetData();
            Output = new FlattenedImage(NumberOfUnits);
            if(Units == null)
            {
                Units = new Unit[NumberOfUnits];
                for(int i = 0; i < NumberOfUnits; i++)
                {
                    Units[i] = new Unit(previous.Size);
                }
            }
        }
    }
}
