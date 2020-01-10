using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    public class ConvolutionalNeuralNetwork
    {
        public InputLayer Input { get; }
        public List<NetworkLayer> NetworkLayers { get; }

        public string ColorScheme { get; set; }
        public ConvolutionalNeuralNetwork(string modelFile)
        {
            ConvolutionalNeuralNetwork item = LoadJson(modelFile);
            Input = item.Input;
            NetworkLayers = item.NetworkLayers;

            ColorScheme = item.ColorScheme;
        }

        [JsonConstructor]
        public ConvolutionalNeuralNetwork(InputLayer input, List<NetworkLayer> networkLayers, string colorScheme)
        {
            Input = input;
            NetworkLayers = networkLayers;

            ColorScheme = colorScheme;
        }

        public ConvolutionalNeuralNetwork(int inputSize, string colorScheme)
        {
            Input = new InputLayer(inputSize, colorScheme);
            NetworkLayers = new List<NetworkLayer>();

            ColorScheme = colorScheme;
        }

        public void Add(NetworkLayer layer)
        {
            NetworkLayers.Add(layer);
        }

        private double GetError(double x, double y)
        {
            return 1.0 / 2.0 * (x - y) * (x - y);
        }

        private FlattenedImage[] GetCrossentropyLoss(double[] output, char c)
        {
            FlattenedImage[] result = new FlattenedImage[output.Length];

            int cInt = (int)(c - 65);
            //int cInt = (int)(c - 48);

            for (int i = 0; i < output.Length; i++)
            {
                if(i == cInt)
                {
                    double log = -1.0 / output[i];
                    result[i] = new FlattenedImage(1, new double[1] { log });
                }
                else
                {
                    result[i] = new FlattenedImage(1, new double[1] { 0 });
                }
            }

            return result;
        }



        private FlattenedImage[] GetCrossentropyError(double[] output, double[] expected, char c)
        {
            FlattenedImage[] result = new FlattenedImage[output.Length];

            double[] expectedExp = MatrixUtils.GetExp(expected);
            double expectedSum = expected.Sum();

            //int cInt = (int)(c - 65);

            //for (int i = 0; i < output.Length; i++)
            //{
            //    if (i == cInt)
            //    {
            //        double log = -Math.Log(output[i]);
            //        result[i] = new FlattenedImage(1, new double[1] { log });
            //    }
            //    else
            //    {
            //        result[i] = new FlattenedImage(1, new double[1] { 0 });
            //    }
            //}

            for (int i = 0; i < output.Length; i++)
            {
                result[i] = new FlattenedImage(1, new double[1] { expectedExp[i]/expectedSum - output[i] });
            }

            return result;
        }

        private FlattenedImage[] GetErrorArray(double[] actualOutput, double[] expectedOutput)
        {
            FlattenedImage[] result = new FlattenedImage[actualOutput.Length];

            for(int i = 0; i < actualOutput.Length; i++)
            {
                double[] value = new double[1];
                //value[0] = GetError(actualOutput[i], expectedOutput[i]);
                double d = actualOutput[i] - expectedOutput[i];
                value[0] = 1.0/2.0 * (d * d);
                result[i] = new FlattenedImage(1, value);
            }

            return result;
        }
        public void Backpropagate(double[] actualOutput, double[] expected, char outputChar, double learningRate)
        {
            //FlattenedImage[] error = GetErrorArray(actualOutput, expected);
            FlattenedImage[] error = GetCrossentropyLoss(actualOutput, outputChar);

            LayerOutput[] nextError = ((DenseLayer)NetworkLayers[NetworkLayers.Count - 1]).Backpropagate(error, learningRate, outputChar - 65);
            //LayerOutput[] nextError = NetworkLayers[NetworkLayers.Count - 1].Backpropagate(error, learningRate);

            for (int i = NetworkLayers.Count - 2; i >= 0; i--)
            {
                nextError = NetworkLayers[i].Backpropagate(nextError, learningRate);
            }
        }

        public void Train(List<InputOutputPair> trainingSet, double learningRate)
        {
            for (int image = 0; image < trainingSet.Count; image++)
            {
                FilteredImage input;
                if (ColorScheme == "rgb")
                {
                    input = ImageProcessing.GetNormalizedFilteredImage(new Bitmap(trainingSet[image].Input));
                }
                else
                {
                    input = ImageProcessing.GetNormalizedGrayscaleFilteredImage(new Bitmap(trainingSet[image].Input));
                }

                double[] actualOutput = RecogniseImage(input);

                Backpropagate(actualOutput, trainingSet[image].Output, trainingSet[image].OutputChar, learningRate);

                if(Math.Abs(((ConvolutionalLayer)NetworkLayers[0]).Filters[0].Kernels[0].ElementSum) > 1000)
                {
                    throw new Exception("Weights over 1000");
                }

                if (Double.IsNaN(((ConvolutionalLayer)NetworkLayers[0]).Filters[0].Kernels[0].Values[0,0]))
                {
                    throw new Exception("Nan error");
                }
            }
        }

        public void TrainMnist(InputOutputPair pair, double learningRate)
        {
            FilteredImage input;
                if (ColorScheme == "rgb")
                {
                    input = ImageProcessing.GetNormalizedFilteredImage(new Bitmap(pair.Input));
                }
                else
                {
                    //input = ImageProcessing.GetNormalizedGrayscaleFilteredImage(new Bitmap(pair.Input));
                    input = ImageProcessing.GetNormalizedMnist(new Bitmap(pair.Input));
                }

                double[] actualOutput = RecogniseImage(input);

                Backpropagate(actualOutput, pair.Output, pair.OutputChar, learningRate);

                if (Math.Abs(((ConvolutionalLayer)NetworkLayers[0]).Filters[0].Kernels[0].ElementSum) > 1000)
                {
                    throw new Exception("Weights over 1000");
                }

                if (Double.IsNaN(((ConvolutionalLayer)NetworkLayers[0]).Filters[0].Kernels[0].Values[0, 0]))
                {
                    throw new Exception("Nan error");
                }
        }

        private char GetReconisedChar(double[] output)
        {
            int maxi = 0;

            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > output[maxi])
                {
                    maxi = i;
                }
            }

            return (char)('A' + maxi);
        }

        public double Evaluate(List<InputOutputPair> set)
        {
            double error = 0;
            int correct = 0, total = 0;

            int N = set.Count;

            for(int i = 0; i < N; i++)
            {
                InputOutputPair pair = set[i];

                FilteredImage input;
                if (ColorScheme == "rgb")
                {
                    input = ImageProcessing.GetNormalizedFilteredImage(new Bitmap(pair.Input));
                }
                else
                {
                    input = ImageProcessing.GetNormalizedGrayscaleFilteredImage(new Bitmap(pair.Input));
                }

                double[] actualOutput = RecogniseImage(input);

                error += LeastSquaredError(actualOutput, pair.Output);

                total++;

                if (GetReconisedChar(actualOutput) == pair.OutputChar)
                {
                    correct++;
                }

                int expected = pair.OutputChar - 65;
                int obtained = GetReconisedChar(actualOutput) - 65;

                total++;
            }


            return (double)correct / total;
        }

        public EvaluationMetrics EvaluateMetrics(List<InputOutputPair> set)
        {
            //double error = 0;
            //int correct = 0, total = 0;

            int[,] matrix = new int[26, 26];
            int total = 0;

            int N = set.Count;

            for (int i = 0; i < N; i++)
            {
                InputOutputPair pair = set[i];

                FilteredImage input;
                if (ColorScheme == "rgb")
                {
                    input = ImageProcessing.GetNormalizedFilteredImage(new Bitmap(pair.Input));
                }
                else
                {
                    input = ImageProcessing.GetNormalizedGrayscaleFilteredImage(new Bitmap(pair.Input));
                }

                double[] actualOutput = RecogniseImage(input);

                //error += LeastSquaredError(actualOutput, pair.Output);

                //total++;

                //if (GetReconisedChar(actualOutput) == pair.OutputChar)
                //{
                //    correct++;
                //}

                int expected = pair.OutputChar - 65;
                int obtained = GetReconisedChar(actualOutput) - 65;

                matrix[expected, obtained]++;

                total++;
            }


            return new EvaluationMetrics(matrix, total);
        }

        public EvaluationMetrics EvaluateParallel(List<InputOutputPair> set)
        {
            int[,] matrix = new int[26, 26];

            int N = 26;

            int total = set.Count;

            Task[] tasks = new Task[N];

            int chunkSize = (N + total - 1) / N;

            for(int t = 0; t < N; t++)
            {
                int start = t * chunkSize;
                int end = Math.Min(start + chunkSize, total);
                tasks[t] = Task.Run(() =>
                {
                    for (int i = start; i < end; i++)
                    {
                        InputOutputPair pair = set[i];

                        FilteredImage input;
                        if (ColorScheme == "rgb")
                        {
                            input = ImageProcessing.GetNormalizedFilteredImage(new Bitmap(pair.Input));
                        }
                        else
                        {
                            input = ImageProcessing.GetNormalizedGrayscaleFilteredImage(new Bitmap(pair.Input));
                        }

                        double[] actualOutput = RecogniseImage(input);

                        int expected = pair.OutputChar - 65;
                        int obtained = GetReconisedChar(actualOutput) - 65;

                        Monitor.Enter(matrix);
                        matrix[expected, obtained]++;
                        Monitor.Exit(matrix);
                    }
                });
            }

            Task.WaitAll(tasks);

            return new EvaluationMetrics(matrix, total);
        }

        private double LeastSquaredError(double[] actual, double[] output)
        {
            double sum = 0;

            for(int i = 0; i < actual.Length; i++)
            {
                sum += (actual[i] - output[i]) * (actual[i] - output[i]);
            }

            return Math.Sqrt(sum);
        }

        private double[] GetOutput()
        {
            FlattenedImage result = (FlattenedImage)NetworkLayers[NetworkLayers.Count - 1].GetData();
            return result.Values;
        }

        public double[] RecogniseImage(FilteredImage image)
        {
            Input.SetInputImage(image);

            foreach (NetworkLayer layer in NetworkLayers)
            {
                layer.ComputeOutput();
            }

            return GetOutput();
        }

        private ConvolutionalNeuralNetwork LoadJson(string modelJson)
        {
            using (StreamReader r = File.OpenText(modelJson))
            {
                string json = r.ReadToEnd();
                ConvolutionalNeuralNetwork item = JsonConvert.DeserializeObject<ConvolutionalNeuralNetwork>(json);

                return item;
            }
        }

        public void Compile()
        {
            Input.CompileLayer(null);
            if (NetworkLayers.Count == 0)
            {
                throw new Exception("No layers added.");
            }
            else
            {
                NetworkLayers[0].CompileLayer(Input);
                for (int i = 1; i < NetworkLayers.Count; i++)
                {
                    NetworkLayers[i].CompileLayer(NetworkLayers[i - 1]);
                }
            }
        }
    }
}
