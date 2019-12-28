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
        public OutputLayer Output { get; }
        public ConvolutionalNeuralNetwork(string modelFile, string weightsFile)
        {
            ConvolutionalNeuralNetwork item = LoadJson(modelFile);
            Input = item.Input;
            NetworkLayers = item.NetworkLayers;
            Output = item.Output;
        }

        [JsonConstructor]
        public ConvolutionalNeuralNetwork(InputLayer input, List<NetworkLayer> networkLayers, OutputLayer output)
        {
            Input = input;
            NetworkLayers = networkLayers;
            Output = output;
        }

        public ConvolutionalNeuralNetwork(int inputWidth, int inputHeight, int outputSize)
        {
            Input = new InputLayer(inputWidth, inputHeight);
            Output = new OutputLayer(outputSize);
            NetworkLayers = new List<NetworkLayer>();
        }

        public void Add(NetworkLayer layer)
        {
            NetworkLayers.Add(layer);
        }

        public void Train()
        {

        }

        public void Evaluate()
        {

        }

        public char RecogniseImage(string path)
        {
            Input.SetInputImage(path);

            foreach(NetworkLayer layer in NetworkLayers)
            {
                layer.ComputeOutput();
            }


            return 'A';
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
                Output.PreviousLayer = Input;
            }
            else
            {
                NetworkLayers[0].CompileLayer(Input);
                for (int i = 1; i < NetworkLayers.Count; i++)
                {
                    NetworkLayers[i].CompileLayer(NetworkLayers[i - 1]);
                }
                Output.CompileLayer(NetworkLayers[NetworkLayers.Count - 1]);
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
