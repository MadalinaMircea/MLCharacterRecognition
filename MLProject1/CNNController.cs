using HDF5DotNet;
using MLProject1.CNN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1
{
    public class CNNController
    {
        ConvolutionalNeuralNetwork model;
        ImageController Ctrl;
        public void CreateAndCompileModel()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();
            SigmoidActivation sigmoidActivation = new SigmoidActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "rgb");
            model.Add(new ConvolutionalLayer(32, 5, reluActivation, "same"));
            model.Add(new MaxPoolingLayer());
            model.Add(new ConvolutionalLayer(64, 3, reluActivation, "same"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new ConvolutionalLayer(128, 3, reluActivation, "same"));
            model.Add(new FlattenLayer());
            model.Add(new DenseLayer(512, sigmoidActivation));
            model.Add(new DropoutLayer(0.5));
            model.Add(new DenseLayer(26, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModel2()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();
            SigmoidActivation sigmoidActivation = new SigmoidActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "grayscale");
            model.Add(new ConvolutionalLayer(5, 5, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new ConvolutionalLayer(5, 3, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new FlattenLayer());
            model.Add(new DropoutLayer(0.5));
            model.Add(new DenseLayer(26, sigmoidActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModel3()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize,"rgb");
            model.Add(new ConvolutionalLayer(5, 5, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new ConvolutionalLayer(5, 3, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new FlattenLayer());
            model.Add(new DropoutLayer(0.5));
            model.Add(new DenseLayer(26, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModel4()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "rgb");
            model.Add(new ConvolutionalLayer(5, 5, reluActivation, "same"));
            model.Add(new MaxPoolingLayer());
            model.Add(new ConvolutionalLayer(5, 3, reluActivation, "same"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new FlattenLayer());
            model.Add(new DropoutLayer(0.5));
            model.Add(new DenseLayer(26, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModel5()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "grayscale");
            model.Add(new ConvolutionalLayer(8, 5, reluActivation, "same"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new FlattenLayer());
            model.Add(new DenseLayer(26, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModel6()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "grayscale");
            model.Add(new ConvolutionalLayer(8, 5, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new FlattenLayer());
            model.Add(new DenseLayer(26, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModelMnist()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 28;

            NoActivation noActivation = new NoActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "grayscale");
            model.Add(new ConvolutionalLayer(8, 3, noActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new FlattenLayer());
            model.Add(new DenseLayer(10, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
        }

        public void CreateAndCompileModel(string jsonPath, string weightsDirectory)
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            model = new ConvolutionalNeuralNetwork(jsonPath);

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
            Console.WriteLine("Reading weights");
            //ReadWeightsFromDirectory(weightsDirectory);

            ReadWeightsFromDirectory(weightsDirectory);
        }

        public char RecogniseImage(Bitmap img)
        {
            double[] resultProbs = model.RecogniseImage(ImageProcessing.GetNormalizedGrayscaleFilteredImage(img));
            int maxi = -1;
            double maxx = -1;

            for (int i = 0; i < resultProbs.Length; i++)
            {
                if (resultProbs[i] > maxx)
                {
                    maxx = resultProbs[i];
                    maxi = i;
                }
            }

            return (char)('A' + maxi);
        }

        public EvaluationMetrics EvaluateMetrics(List<InputOutputPair> set)
        {
            Console.WriteLine("Evaluating");
            //return model.EvaluateMetrics(set);
            return model.EvaluateParallel(set);
        }

        public double Evaluate(List<InputOutputPair> set)
        {
            Console.WriteLine("Evaluating");
            return model.Evaluate(set);
            //return model.EvaluateParallel(set);
        }

        public void Test(int process)
        {
            EvaluationMetrics metrics = EvaluateMetrics(Ctrl.Repo.TestingSetPaths);

            Console.WriteLine("Process " + process + " Testing Accuracy: " + metrics.OverallAccuracy);
        }

        public void Train(int batchSize, string weightDirectory, int process, double learningRate)
        {
            int batchNumber = Ctrl.Repo.TrainingSetPaths.Count / batchSize;

            double accuracy = 0.27;

            for (int batch = 0; batch < batchNumber; batch++)
                {
                    int startPos = batch * batchSize;

                    Console.WriteLine("Process " + process + " Training batch: " + batch);
                    model.Train(Ctrl.Repo.TrainingSetPaths.Skip(startPos).Take(batchSize).ToList(), learningRate);
                    double modelAccuracy = Evaluate(Ctrl.Repo.ValidationSetPaths);
                    if (modelAccuracy > accuracy)
                    {
                        WriteWeightsToDirectory(weightDirectory);
                        accuracy = modelAccuracy;
                    }
                    Console.WriteLine("Process " + process + " Accuracy: " + modelAccuracy);
                    if (modelAccuracy >= 0.90)
                    {
                        WriteWeightsToDirectory("bestWeights" + process);
                        Console.WriteLine("Process " + process + " Accuracy over 90% achieved!");

                    modelAccuracy = Evaluate(Ctrl.Repo.TestingSetPaths);
                        if(modelAccuracy >= 0.90)
                        {
                            Console.WriteLine("Process " + process + " Testing accuracy over 90% achieved!");
                            break;
                        }
                    }

                    if (batch % 50 == 0 && batch > 0)
                    {
                    modelAccuracy = Evaluate(Ctrl.Repo.TestingSetPaths);
                        Console.WriteLine("Process " + process + " Testing accuracy: " + modelAccuracy);
                    }
                }
        }

        public void TrainOneMnist(double learningRate)
        {
            InputOutputPair pair = new InputOutputPair("E:\\Programs\\MNIST-JPG-master\\output\\training\\0\\1.jpg",
                "E:\\Programs\\MNIST-JPG-master\\output\\training\\0");
            model.TrainMnist(pair, learningRate);
        }

        public void WriteToFile(string modelFile, string weightsDirectory)
        {
            Task t = Task.Run(() =>
            {
                WriteWeightsToDirectory(weightsDirectory);
            });

            string json = JsonConvert.SerializeObject(model, Formatting.Indented);
            File.WriteAllText(modelFile, json);

            Console.WriteLine("Model written to file");

            t.Wait();
        }

        public void WriteWeightsToDirectory(string weightsDirectory)
        {
            Task[] tasks = new Task[model.NetworkLayers.Count];

            for(int i = 0; i < model.NetworkLayers.Count; i++)
            {
                int taski = 0 + i;

                tasks[taski] = Task.Run(() =>
                {
                    string layerPath = weightsDirectory + "\\" + model.NetworkLayers[taski].Type + taski;

                    switch (model.NetworkLayers[taski].Type)
                    {
                        case "Convolutional":
                            Directory.CreateDirectory(layerPath);
                            ConvolutionalLayer auxLayer = (ConvolutionalLayer)model.NetworkLayers[taski];

                            for (int filter = 0; filter < auxLayer.FilterNumber; filter++)
                            {
                                int taskf = 0 + filter;

                                File.WriteAllText(layerPath + "\\Filter" + filter + ".json", JsonConvert.SerializeObject(auxLayer.Filters[taskf]));

                            }
                            break;
                        case "Dense":
                            Directory.CreateDirectory(layerPath);
                            DenseLayer auxDense = (DenseLayer)model.NetworkLayers[taski];
                            for (int unit = 0; unit < auxDense.NumberOfUnits; unit++)
                            {
                                File.WriteAllText(layerPath + "\\Unit" + unit + ".json", JsonConvert.SerializeObject(auxDense.Units[unit]));
                            }
                            break;
                        default:
                            break;
                    }
                });
            }

            Task.WaitAll(tasks);

            Console.WriteLine("Weights written to file");
        }

        public void ReadWeightsFromDirectory(string weightsDirectory)
        {
            Task[] tasks = new Task[model.NetworkLayers.Count];

            for (int i = 0; i < model.NetworkLayers.Count; i++)
            {
                int taski = 0 + i;

                tasks[taski] = Task.Run(() =>
                {
                    string layerPath = weightsDirectory + "\\" + model.NetworkLayers[taski].Type + taski;

                    switch (model.NetworkLayers[taski].Type)
                    {
                        case "Convolutional":
                            ConvolutionalLayer auxLayer = (ConvolutionalLayer)model.NetworkLayers[taski];
                            for (int filter = 0; filter < auxLayer.FilterNumber; filter++)
                            {
                                string filterPath = layerPath + "\\Filter" + filter + ".json";
                                string json = File.ReadAllText(filterPath);
                                auxLayer.Filters[filter] = JsonConvert.DeserializeObject<Filter>(json);
                            }
                            break;
                        case "Dense":
                            DenseLayer auxDense = (DenseLayer)model.NetworkLayers[taski];
                            for (int unit = 0; unit < auxDense.NumberOfUnits; unit++)
                            {
                                string json = File.ReadAllText(layerPath + "\\Unit" + unit + ".json");
                                auxDense.Units[unit] = JsonConvert.DeserializeObject<Unit>(json);
                            }
                            break;
                        default:
                            break;
                    }
                });
            }

            Task.WaitAll(tasks);
        }
        public void PrepareImageSets(string trainingPath, string testingPath, string validationPath)
        {
            Console.WriteLine("Preparing dataset");
            ImageRepository Repo = new ImageRepository();
            Ctrl = new ImageController(Repo);

            Task t1 = Task.Run(() => { Ctrl.ReadSet("train", trainingPath); });
            Task t2 = Task.Run(() => { Ctrl.ReadSet("test", testingPath); });
            Task t3 = Task.Run(() => { Ctrl.ReadSet("valid", validationPath); });

            t1.Wait();
            t2.Wait();
            t3.Wait();


            //Ctrl.ReadSet("train", trainingPath);
            //Ctrl.ReadSet("test", testingPath);
            //Ctrl.ReadSet("valid", validationPath);

            Console.WriteLine("Shuffling sets");
            ShuffleSets();
        }

        public void ShuffleSets()
        {
            Ctrl.ShuffleSets();
        }

        public void EvaluateModel()
        {
            EvaluationMetrics training = EvaluateMetrics(Ctrl.Repo.TrainingSetPaths);
            File.WriteAllText("TrainingPerformance.json", JsonConvert.SerializeObject(training));

            EvaluationMetrics testing = EvaluateMetrics(Ctrl.Repo.TestingSetPaths);
            File.WriteAllText("TestingPerformance.json", JsonConvert.SerializeObject(testing));

            EvaluationMetrics validation = EvaluateMetrics(Ctrl.Repo.ValidationSetPaths);
            File.WriteAllText("ValidationPerformance.json", JsonConvert.SerializeObject(validation));
        }
    }
}
