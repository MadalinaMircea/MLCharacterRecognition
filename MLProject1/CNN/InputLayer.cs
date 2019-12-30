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
    class InputLayer : NetworkLayer
    {
        public int Size { get; set; }

        [JsonIgnore]
        public FilteredImage Output { get; set; }

        public InputLayer(int size) : base("Input")
        {
            Size = size;
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
                Output = new FilteredImage(3, Size);
            }
        }
    }
}
