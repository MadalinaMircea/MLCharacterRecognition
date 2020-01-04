using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
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
            else if (activationFunction == "sigmoid")
            {
                ActivationFunction = new SigmoidActivation();
            }
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void ComputeOutput()
        {
            FlattenedImage previous = (FlattenedImage)PreviousLayer.GetData();

            Task[] tasks = new Task[NumberOfUnits];

            for (int i = 0; i < NumberOfUnits; i++)
            {
                int taski = 0 + i;

                tasks[taski] = Task.Run(() =>
                {
                    Output.Values[taski] = Units[taski].ComputeOutput(previous);
                });
            }

            Task.WaitAll(tasks);

            Output = ActivationFunction.Activate(Output);
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

        public override LayerOutput[] Backpropagate(LayerOutput[] nextOutput, double learningRate)
        {
            int weightsPerUnit = Units[0].NumberOfWeights;
            FlattenedImage[] result = new FlattenedImage[weightsPerUnit];
            FlattenedImage previous = (FlattenedImage)PreviousLayer.GetData();

            for(int i = 0; i < weightsPerUnit; i++)
            {
                result[i] = new FlattenedImage(NumberOfUnits);
            }

            FlattenedImage activationDerivative = ActivationFunction.GetDerivative(Output);

            Task[] tasks = new Task[NumberOfUnits];

            for (int unit = 0; unit < NumberOfUnits; unit++)
            {
                int tasku = 0 + unit;

                tasks[tasku] = Task.Run(() =>
                {
                    Unit unitAux = Units[tasku];

                    FlattenedImage nextErrors = (FlattenedImage)nextOutput[tasku];

                    double unitSum = nextErrors.Values.Sum();
                    double unitDerivative = unitSum * activationDerivative.Values[tasku];

                    for (int weight = 0; weight < unitAux.NumberOfWeights; weight++)
                    {
                        Monitor.Enter(result);
                        result[weight].Values[tasku] = unitDerivative * unitAux.Weights[weight];
                        Monitor.Exit(result);
                        double deltaW = unitDerivative * previous.Values[weight];
                        unitAux.Weights[weight] -= learningRate * deltaW;
                    }
                });
            }

            Task.WaitAll(tasks);

            //for (int unit = 0; unit < NumberOfUnits; unit++)
            //{
            //    Unit unitAux = Units[unit];

            //        FlattenedImage nextErrors = (FlattenedImage)nextOutput[unit];

            //        double unitSum = nextErrors.Sum();
            //        double unitDerivative = unitSum * activationDerivative.Values[unit];

            //        for (int weight = 0; weight < unitAux.NumberOfWeights; weight++)
            //        {
            //            Monitor.Enter(result);
            //            result[weight].Values[unit] = unitDerivative * unitAux.Weights[weight];
            //            Monitor.Exit(result);
            //            double deltaW = unitDerivative * previous.Values[weight];
            //            unitAux.Weights[weight] -= learningRate * deltaW;
            //        }
            //}

            return result;
        }

        public LayerOutput[] Backpropagate(LayerOutput[] nextOutput, double learningRate, int correctClass)
        {
            int weightsPerUnit = Units[0].NumberOfWeights;
            FlattenedImage[] result = new FlattenedImage[weightsPerUnit];
            FlattenedImage previous = (FlattenedImage)PreviousLayer.GetData();

            for (int i = 0; i < weightsPerUnit; i++)
            {
                result[i] = new FlattenedImage(NumberOfUnits);
            }

            FlattenedImage activationDerivative = ActivationFunction.GetDerivative(Output, correctClass);

            Task[] tasks = new Task[NumberOfUnits];

            for (int unit = 0; unit < NumberOfUnits; unit++)
            {
                int tasku = 0 + unit;

                tasks[tasku] = Task.Run(() =>
                {
                    Unit unitAux = Units[tasku];

                    FlattenedImage nextErrors = (FlattenedImage)nextOutput[tasku];

                    double unitSum = nextErrors.Values.Sum();
                    double unitDerivative = unitSum * activationDerivative.Values[tasku];

                    for (int weight = 0; weight < unitAux.NumberOfWeights; weight++)
                    {
                        Monitor.Enter(result);
                        result[weight].Values[tasku] = unitDerivative * unitAux.Weights[weight];
                        Monitor.Exit(result);
                        double deltaW = unitDerivative * previous.Values[weight];
                        unitAux.Weights[weight] -= learningRate * deltaW;
                    }
                });
            }

            Task.WaitAll(tasks);

            //for (int unit = 0; unit < NumberOfUnits; unit++)
            //{
            //    Unit unitAux = Units[unit];

            //    double unitSum = nextOutput[unit].Sum();
            //    double unitDerivative = unitSum * activationDerivative.Values[unit];

            //    for (int weight = 0; weight < unitAux.NumberOfWeights; weight++)
            //    {
            //        Monitor.Enter(result);
            //        result[weight].Values[unit] = unitDerivative * unitAux.Weights[weight];
            //        Monitor.Exit(result);
            //        double deltaW = unitDerivative * previous.Values[weight];
            //        unitAux.Weights[weight] -= learningRate * deltaW;
            //    }
            //}

            return result;
        }
    }
}
