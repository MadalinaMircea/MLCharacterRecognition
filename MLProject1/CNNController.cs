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
    class CNNController
    {
        ConvolutionalNeuralNetwork model;
        ImageRepository Repo;
        public void CreateAndCompileModel()
        {
            int imgSize = 75;

            ReluActivation reluActivation = new ReluActivation();
            SoftmaxActivation softmaxActivation = new SoftmaxActivation();

            model = new ConvolutionalNeuralNetwork(imgSize);
            model.Add(new ConvolutionalLayer(32, 7, reluActivation));
            model.Add(new MaxPoolingLayer());
            model.Add(new ConvolutionalLayer(64, 5, reluActivation));
            model.Add(new MaxPoolingLayer());
            model.Add(new DropoutLayer(0.2));
            model.Add(new ConvolutionalLayer(128, 5, reluActivation));
            model.Add(new FlattenLayer());
            model.Add(new DenseLayer(512, reluActivation));
            model.Add(new DropoutLayer(0.5));
            model.Add(new DenseLayer(26, softmaxActivation));

            model.Compile();

            string json = JsonConvert.SerializeObject(model, Formatting.Indented);
            File.WriteAllText("tryNewJson.json", json);
        }

        public void CreateAndCompileModel(string jsonPath, string weightsDirectory)
        {
            model = new ConvolutionalNeuralNetwork(jsonPath, "");

            model.Compile();

            ReadWeightsFromDirectory(weightsDirectory);
        }

        public char RecogniseImage(Bitmap img)
        {
            double[] resultProbs = model.RecogniseImage(ImageProcessing.GetNormalizedFilteredImage(img));
            int maxi = -1;
            double maxx = -1;

            for(int i = 0; i < resultProbs.Length; i++)
            {
                if(resultProbs[i] > maxx)
                {
                    maxx = resultProbs[i];
                    maxi = i;
                }
            }

            return (char)('A' + maxi);
        }

        public void WriteToFile(string modelFile, string weightsDirectory)
        {
            string json = JsonConvert.SerializeObject(model, Formatting.Indented);
            File.WriteAllText(modelFile, json);

            WriteWeightsToDirectory(weightsDirectory);
        }

        public void WriteWeightsToDirectory(string weightsDirectory)
        {
            for(int i = 0; i < model.NetworkLayers.Count; i++)
            {
                string layerPath = weightsDirectory + "\\" + model.NetworkLayers[i].Type + i;
                
                switch(model.NetworkLayers[i].Type)
                {
                    case "Convolutional":
                        Directory.CreateDirectory(layerPath);
                        ConvolutionalLayer auxLayer = (ConvolutionalLayer)model.NetworkLayers[i];
                        for (int filter = 0; filter < auxLayer.FilterNumber; filter++)
                        {
                            string filterPath = layerPath + "\\Filter" + filter;
                            Directory.CreateDirectory(filterPath);
                            Filter auxFilter = auxLayer.Filters[filter];

                            for(int kernel = 0; kernel < auxFilter.KernelNumber; kernel++)
                            {
                                File.WriteAllText(filterPath + "\\Kernel" + kernel + ".json", JsonConvert.SerializeObject(auxFilter.Kernels[kernel]));
                            }
                        }
                        break;
                    case "Dense":
                        Directory.CreateDirectory(layerPath);
                        DenseLayer auxDense = (DenseLayer)model.NetworkLayers[i];
                        for(int unit = 0; unit < auxDense.NumberOfUnits; unit++)
                        {
                            File.WriteAllText(layerPath + "\\Unit" + unit + ".json", JsonConvert.SerializeObject(auxDense.Units[unit]));
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        public void ReadWeightsFromDirectory(string weightsDirectory)
        {
            for (int i = 0; i < model.NetworkLayers.Count; i++)
            {
                string layerPath = weightsDirectory + "\\" + model.NetworkLayers[i].Type + i;

                switch (model.NetworkLayers[i].Type)
                {
                    case "Convolutional":
                        ConvolutionalLayer auxLayer = (ConvolutionalLayer)model.NetworkLayers[i];
                        for (int filter = 0; filter < auxLayer.FilterNumber; filter++)
                        {
                            string filterPath = layerPath + "\\Filter" + filter;
                            Filter auxFilter = auxLayer.Filters[filter];

                            for (int kernel = 0; kernel < auxFilter.KernelNumber; kernel++)
                            {
                                string json = File.ReadAllText(filterPath + "\\Kernel" + kernel + ".json");
                                auxFilter.Kernels[kernel] = JsonConvert.DeserializeObject<Kernel>(json);
                            }
                        }
                        break;
                    case "Dense":
                        DenseLayer auxDense = (DenseLayer)model.NetworkLayers[i];
                        for (int unit = 0; unit < auxDense.NumberOfUnits; unit++)
                        {
                            string json = File.ReadAllText(layerPath + "\\Unit" + unit + ".json");
                            auxDense.Units[unit] = JsonConvert.DeserializeObject<Unit>(json);
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        public void PrepareImageSets(string trainingPath, string testingPath, string validationPath)
        {
            Repo = new ImageRepository();
            ImageController ctrl = new ImageController(Repo);

            Task t1 = Task.Run(() => { ctrl.ReadSet("train", trainingPath); });
            Task t2 = Task.Run(() => { ctrl.ReadSet("test", testingPath); });
            Task t3 = Task.Run(() => { ctrl.ReadSet("valid", validationPath); });

            t1.Wait();
            t2.Wait();
            t3.Wait();

            ctrl.ShuffleSets();
        }
    }
}
