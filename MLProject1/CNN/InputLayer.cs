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
        public int Width { get; set; }
        public int Height { get; set; }

        //public RGBPixel[,] Image { get; set; }

        public FilteredImage Output { get; set; }

        public InputLayer(int width, int height, FilteredImage output = null) : base("Input")
        {
            Width = width;
            Height = height;
            Output = output;
        }

        //public InputLayer(int width, int height, RGBPixel[,] image)
        //{
        //    Width = width;
        //    Height = height;
        //    Image = image;
        //}

        public InputLayer(int width, int height, string imagePath) : base("Input")
        {
            Width = width;
            Height = height;
            SetInputImage(imagePath);
        }

        public InputLayer(int width, int height, Bitmap bitmap) : base("Input")
        {
            Width = width;
            Height = height;
            SetInputImage(bitmap);
        }

        public void SetInputImage(string imagePath)
        {
            //Image = ImageProcessing.GetNormalizedRGBMatrix(imagePath);
            SetInputImage(new Bitmap(imagePath));
        }

        public void SetInputImage(Bitmap bitmap)
        {
            //Image = ImageProcessing.GetNormalizedRGBMatrix(bitmap);
            Output = ImageProcessing.GetNormalizedFilteredImage(bitmap);
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
                Output = new FilteredImage(3, new FilteredImageChannel[3]);
            }
        }
    }
}
