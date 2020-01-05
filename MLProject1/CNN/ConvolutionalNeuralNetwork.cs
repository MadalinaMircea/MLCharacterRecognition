﻿using Newtonsoft.Json;
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
    class ConvolutionalNeuralNetwork
    {
        public InputLayer Input { get; }
        public List<NetworkLayer> NetworkLayers { get; }

        public string ColorScheme { get; set; }
        public ConvolutionalNeuralNetwork(string modelFile, string weightsFile, string colorScheme = "rgb")
        {
            ConvolutionalNeuralNetwork item = LoadJson(modelFile);
            Input = item.Input;
            NetworkLayers = item.NetworkLayers;

            ColorScheme = colorScheme;
        }

        [JsonConstructor]
        public ConvolutionalNeuralNetwork(InputLayer input, List<NetworkLayer> networkLayers, string colorScheme = "rgb")
        {
            Input = input;
            NetworkLayers = networkLayers;

            ColorScheme = colorScheme;
        }

        public ConvolutionalNeuralNetwork(int inputSize, string colorScheme = "rgb")
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

            for (int i = 0; i < output.Length; i++)
            {
                if(i == cInt)
                {
                    double log = -Math.Log(output[i]);
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
                value[0] = actualOutput[i] - expectedOutput[i];
                result[i] = new FlattenedImage(1, value);
            }

            return result;
        }
        public void Backpropagate(double[] actualOutput, double[] expected, char outputChar, double learningRate)
        {
            FlattenedImage[] error = GetErrorArray(actualOutput, expected);
            //FlattenedImage[] error = GetCrossentropyError(actualOutput, expected, outputChar);

            //LayerOutput[] nextError = ((DenseLayer)NetworkLayers[NetworkLayers.Count - 1]).Backpropagate(error, learningRate, outputChar - 65);
            LayerOutput[] nextError = NetworkLayers[NetworkLayers.Count - 1].Backpropagate(error, learningRate);

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

        public EvaluationMetrics Evaluate(List<InputOutputPair> set)
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
            }


            return new EvaluationMetrics(error, (double)correct / total);
        }

        public EvaluationMetrics EvaluateParallel(List<InputOutputPair> set)
        {
            double error = 0;
            int correct = 0, total = 0;

            object o = new object();

            int N = set.Count;

            Task[] tasks = new Task[N];

            for (int i = 0; i < N; i++)
            {
                int taski = 0 + i;

                tasks[i] = Task.Run(() =>
                {
                    InputOutputPair pair = set[taski];

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

                    Monitor.Enter(o);
                    error += LeastSquaredError(actualOutput, pair.Output);

                    total++;

                    if (GetReconisedChar(actualOutput) == pair.OutputChar)
                    {
                        correct++;
                    }
                    Monitor.Exit(o);
                });
            }

            Task.WaitAll(tasks);

            return new EvaluationMetrics(error, (double)correct / total);
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
