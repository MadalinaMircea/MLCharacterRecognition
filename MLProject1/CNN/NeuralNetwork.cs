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
    class NeuralNetwork
    {
        public NeuralInputLayer Input { get; }
        public List<NetworkLayer> NetworkLayers { get; }

        public string ColorScheme { get; set; }
        public NeuralNetwork(string modelFile, string weightsFile, string colorScheme = "rgb")
        {
            NeuralNetwork item = LoadJson(modelFile);
            Input = item.Input;
            NetworkLayers = item.NetworkLayers;

            ColorScheme = colorScheme;
        }

        //private void LoadWeightsFile(string weightsFile)
        //{
        //    H5FileId fileId = H5F.open(weightsFile, H5F.OpenMode.ACC_RDONLY);

        //    H5GroupId groupId = H5G.open(fileId, "/model_weights/conv2d_1/conv2d_1");

        //    H5DataSetId datasetId = H5D.open(groupId, "bias:0");

        //    H5DataTypeId datatypeId = H5D.getType(datasetId);

        //    float[,] arr = new float[32, 1];
        //    H5Array<float> array = new H5Array<float>(arr);

        //    H5D.read<float>(datasetId, datatypeId, array);
        //}

        [JsonConstructor]
        public NeuralNetwork(NeuralInputLayer input, List<NetworkLayer> networkLayers, string colorScheme = "rgb")
        {
            Input = input;
            NetworkLayers = networkLayers;

            ColorScheme = colorScheme;
        }

        public NeuralNetwork(int inputSize, string colorScheme = "rgb")
        {
            Input = new NeuralInputLayer(inputSize, colorScheme);
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
                if (i == cInt)
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

        private FlattenedImage[] GetErrorArray(double[] actualOutput, double[] expectedOutput)
        {
            FlattenedImage[] result = new FlattenedImage[actualOutput.Length];

            for (int i = 0; i < actualOutput.Length; i++)
            {
                double[] value = new double[1];
                //value[0] = GetError(actualOutput[i], expectedOutput[i]);
                value[0] = actualOutput[i] - expectedOutput[i];
                result[i] = new FlattenedImage(1, value);
            }

            return result;
        }
        public void Backpropagate(double[] actualOutput, char outputChar, double learningRate)
        {
            FlattenedImage[] error = GetCrossentropyLoss(actualOutput, outputChar);

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
                FlattenedImage input;
                if (ColorScheme == "rgb")
                {
                    input = ImageProcessing.GetNormalizedFlattenedImage(new Bitmap(trainingSet[image].Input));
                }
                else
                {
                    input = ImageProcessing.GetNormalizedGrayscaleFlattenedImage(new Bitmap(trainingSet[image].Input));
                }

                double[] actualOutput = RecogniseImage(input);

                Backpropagate(actualOutput, trainingSet[image].OutputChar, learningRate);
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

            for (int i = 0; i < N; i++)
            {
                InputOutputPair pair = set[i];

                FlattenedImage input;
                if (ColorScheme == "rgb")
                {
                    input = ImageProcessing.GetNormalizedFlattenedImage(new Bitmap(pair.Input));
                }
                else
                {
                    input = ImageProcessing.GetNormalizedGrayscaleFlattenedImage(new Bitmap(pair.Input));
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
                //int taski = GlobalRandom.GetRandomInt(0, set.Count);

                int taski = 0 + i;

                tasks[i] = Task.Run(() =>
                {
                    InputOutputPair pair = set[taski];

                    FlattenedImage input;
                    if (ColorScheme == "rgb")
                    {
                        input = ImageProcessing.GetNormalizedFlattenedImage(new Bitmap(pair.Input));
                    }
                    else
                    {
                        input = ImageProcessing.GetNormalizedGrayscaleFlattenedImage(new Bitmap(pair.Input));
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

            for (int i = 0; i < actual.Length; i++)
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

        public double[] RecogniseImage(FlattenedImage image)
        {
            Input.SetInputImage(image);

            foreach (NetworkLayer layer in NetworkLayers)
            {
                layer.ComputeOutput();
            }

            //Output.ComputeOutput();

            return GetOutput();
        }

        private NeuralNetwork LoadJson(string modelJson)
        {
            using (StreamReader r = File.OpenText(modelJson))
            {
                string json = r.ReadToEnd();
                NeuralNetwork item = JsonConvert.DeserializeObject<NeuralNetwork>(json);

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
                //Output.CompileLayer(NetworkLayers[NetworkLayers.Count - 1]);
            }
        }
    }
}
