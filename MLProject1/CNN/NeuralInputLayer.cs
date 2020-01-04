using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class NeuralInputLayer : NetworkLayer
    {
        public int Size { get; set; }

        [JsonIgnore]
        public LayerOutput Output { get; set; }

        [JsonIgnore]
        public string ColorScheme { get; set; }

        public NeuralInputLayer(int size, string colorScheme) : base("Input")
        {
            Size = size;

            ColorScheme = colorScheme;
        }

        public void SetInputImage(FlattenedImage image)
        {
            Output = image;
        }

        public override void ComputeOutput()
        {

        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            if (Output == null)
            {
                if (ColorScheme == "rgb")
                {
                    Output = new FlattenedImage(Size);
                }
                else
                {
                    Output = new FlattenedImage(Size);
                }
            }
        }

        public override LayerOutput[] Backpropagate(LayerOutput[] nextOutput, double learningRate)
        {
            throw new NotImplementedException();
        }
    }
}
