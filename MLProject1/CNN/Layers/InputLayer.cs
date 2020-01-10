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
    public class InputLayer : NetworkLayer
    {
        public int Size { get; set; }

        [JsonIgnore]
        public LayerOutput Output { get; set; }

        [JsonIgnore]
        public string ColorScheme { get; set; }

        public InputLayer(int size, string colorScheme) : base("Input")
        {
            Size = size;

            ColorScheme = colorScheme;
        }

        public void SetInputImage(FilteredImage image)
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
                    Output = new FilteredImage(3, Size);
                }
                else
                {
                    Output = new FilteredImage(1, Size);
                }
            }
        }

        public override LayerOutput[] Backpropagate(LayerOutput[] nextOutput, double learningRate)
        {
            throw new NotImplementedException();
        }
    }
}
