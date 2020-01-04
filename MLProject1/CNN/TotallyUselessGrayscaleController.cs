using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class TotallyUselessGrayscaleController
    {
        ConvolutionalNeuralNetwork model;
        ImageRepository Repo;
        public void CreateAndCompileModel4()
        {
            Console.WriteLine("Creating model");
            GlobalRandom.InitializeRandom();

            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize, "grayscale");
            model.Add(new ConvolutionalLayer(5, 3, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new ConvolutionalLayer(5, 3, reluActivation, "valid"));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new FlattenLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new DenseLayer(26, softmaxActivation));

            Console.WriteLine("Model created");

            model.Compile();

            Console.WriteLine("Model compiled");
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

        public EvaluationMetrics Evaluate(List<InputOutputPair> set)
        {
            Console.WriteLine("Evaluating");
            return model.Evaluate(set);
            //return model.EvaluateParallel(set);
        }

        public void Test(int process)
        {
            EvaluationMetrics metrics = Evaluate(Repo.TestingSetPaths);

            Console.WriteLine("Process " + process + " Testing Accuracy: " + metrics.Accuracy + ", Testing Error: " + metrics.Error);
        }

        public void Train(int batchSize, string weightDirectory, int process)
        {
            int batchNumber = Repo.TrainingSetPaths.Count / batchSize;

            double accuracy = 0;

            bool keepGoing = true;

            while (keepGoing)
            {
                for (int batch = 0; batch < batchNumber; batch++)
                {
                    int startPos = batch * batchSize;

                    Console.WriteLine("Process " + process + " Training batch: " + batch);
                    model.Train(Repo.TrainingSetPaths.Skip(startPos).Take(batchSize).ToList(), 0.01);
                    EvaluationMetrics metrics = Evaluate(Repo.ValidationSetPaths);
                    if (metrics.Accuracy > accuracy)
                    {
                        WriteWeightsToDirectory(weightDirectory);
                        accuracy = metrics.Accuracy;
                    }
                    Console.WriteLine("Process " + process + " Accuracy: " + metrics.Accuracy);
                    if (metrics.Accuracy >= 0.90)
                    {
                        WriteWeightsToDirectory("bestWeights" + process);
                        Console.WriteLine("Process " + process + " Accuracy over 90% achieved!");

                        metrics = Evaluate(Repo.TestingSetPaths);
                        if (metrics.Accuracy >= 0.90)
                        {
                            Console.WriteLine("Process " + process + " Testing accuracy over 90% achieved!");
                            keepGoing = false;
                            break;
                        }
                    }

                    if (batch % 50 == 0)
                    {
                        metrics = Evaluate(Repo.TestingSetPaths);
                        Console.WriteLine("Process " + process + " Testing accuracy: " + metrics.Accuracy);
                    }
                }
            }
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

            for (int i = 0; i < model.NetworkLayers.Count; i++)
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

                                //for (int kernel = 0; kernel < auxFilter.KernelNumber; kernel++)
                                //{
                                //    string json = File.ReadAllText(filterPath + "\\Kernel" + kernel + ".json");
                                //    auxFilter.Kernels[kernel] = JsonConvert.DeserializeObject<Kernel>(json);
                                //}
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

        //public void ReadWeightsFromH5(string weightFile)
        //{

        //    //Task[] tasks = new Task[model.NetworkLayers.Count];

        //    int convNr = 1, denseNr = 1;
        //    string initialPath = "/model_weights/";

        //    H5FileId fileId = H5F.open(weightFile, H5F.OpenMode.ACC_RDONLY);

        //    for (int i = 0; i < model.NetworkLayers.Count; i++)
        //    {
        //        int taski = 0 + i;

        //        //tasks[taski] = Task.Run(() =>
        //        //{
        //            switch (model.NetworkLayers[taski].Type)
        //            {
        //                case "Convolutional":
        //                    ConvolutionalLayer auxLayer = (ConvolutionalLayer)model.NetworkLayers[taski];

        //                    string pathAddition = "conv2d_" + convNr;
        //                    string path = initialPath + pathAddition + "/" + pathAddition;

        //                    H5GroupId convGroupId = H5G.open(fileId, path);

        //                    H5DataSetId convDatasetId = H5D.open(convGroupId, "bias:0");

        //                    H5DataTypeId datatypeId = H5D.getType(convDatasetId);

        //                    double[,] arr = new double[auxLayer.FilterNumber, 1];
        //                    H5Array<double> array = new H5Array<double>(arr);

        //                    H5D.read(convDatasetId, datatypeId, array);

        //                    auxLayer.SetBias(arr);

        //                    for (int filter = 0; filter < auxLayer.FilterNumber; filter++)
        //                    {
        //                        convDatasetId = H5D.open(convGroupId, "kernel:0");

        //                        datatypeId = H5D.getType(convDatasetId);

        //                        double[,,,] filterArr = new double[auxLayer.FilterSize, auxLayer.FilterSize,auxLayer.Filters[0].KernelNumber,auxLayer.FilterNumber];
        //                        H5Array<double> filterArray = new H5Array<double>(filterArr);

        //                    H5D.read(convDatasetId, datatypeId, filterArray);

        //                        Filter auxFilter = auxLayer.Filters[filter];

        //                        for (int kernel = 0; kernel < auxFilter.KernelNumber; kernel++)
        //                        {
        //                            //string json = File.ReadAllText(filterPath + "\\Kernel" + kernel + ".json");
        //                            //auxFilter.Kernels[kernel] = JsonConvert.DeserializeObject<Kernel>(json);
        //                        }
        //                    }
        //                    break;
        //                case "Dense":
        //                    //DenseLayer auxDense = (DenseLayer)model.NetworkLayers[taski];
        //                    //for (int unit = 0; unit < auxDense.NumberOfUnits; unit++)
        //                    //{
        //                    //    string json = File.ReadAllText(layerPath + "\\Unit" + unit + ".json");
        //                    //    auxDense.Units[unit] = JsonConvert.DeserializeObject<Unit>(json);
        //                    //}
        //                    break;
        //                default:
        //                    break;
        //            }
        //        //});
        //    }

        //    //Task.WaitAll(tasks);


        //}

        public void PrepareImageSets(string trainingPath, string testingPath, string validationPath)
        {
            Console.WriteLine("Preparing dataset");
            Repo = new ImageRepository();
            ImageController ctrl = new ImageController(Repo);

            Task t1 = Task.Run(() => { ctrl.ReadSet("train", trainingPath); });
            Task t2 = Task.Run(() => { ctrl.ReadSet("test", testingPath); });
            Task t3 = Task.Run(() => { ctrl.ReadSet("valid", validationPath); });

            t1.Wait();
            t2.Wait();
            t3.Wait();

            Console.WriteLine("Shuffling sets");
            ctrl.ShuffleSets();
        }
    }
}
