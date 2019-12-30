using HDF5DotNet;
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
    [Serializable]
    class ConvolutionalNeuralNetwork
    {
        public InputLayer Input { get; }
        public List<NetworkLayer> NetworkLayers { get; }
        public ConvolutionalNeuralNetwork(string modelFile, string weightsFile)
        {
            ConvolutionalNeuralNetwork item = LoadJson(modelFile);
            Input = item.Input;
            NetworkLayers = item.NetworkLayers;
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
        public ConvolutionalNeuralNetwork(InputLayer input, List<NetworkLayer> networkLayers)
        {
            Input = input;
            NetworkLayers = networkLayers;
        }

        public ConvolutionalNeuralNetwork(int inputSize)
        {
            Input = new InputLayer(inputSize);
            NetworkLayers = new List<NetworkLayer>();
        }

        public void Add(NetworkLayer layer)
        {
            NetworkLayers.Add(layer);
        }

        private void TrainModel()
        {

        }

        public void Train(string trainingFolder, string validationFolder)
        {
            foreach(string directoryPath in Directory.GetDirectories(trainingFolder))
            {

            }
        }

        public void Evaluate()
        {

        }

        private double[] GetOutput()
        {
            FlattenedImage result = (FlattenedImage)NetworkLayers[NetworkLayers.Count - 1].GetData();
            return result.Values;
        }

        public double[] RecogniseImage(FilteredImage image)
        {
            Input.SetInputImage(image);

            foreach(NetworkLayer layer in NetworkLayers)
            {
                layer.ComputeOutput();
            }

            //Output.ComputeOutput();

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
                //Output.CompileLayer(NetworkLayers[NetworkLayers.Count - 1]);
            }
        }

        //public override string ToString()
        //{
        //    StringBuilder builder = new StringBuilder("{");
        //    builder.Append(Input.ToString());
        //    builder.Append(",\n[\n\"layers\":\n[");
        //    foreach(NetworkLayer layer in networkLayers)
        //    {
        //        builder.Append(layer.ToString());
        //        builder.Append(",\n");
        //    }
        //    builder.Append("\n],\n");
        //    builder.Append(Output.ToString());
        //    builder.Append("\n}");

        //    return builder.ToString();
        //}
    }
}
