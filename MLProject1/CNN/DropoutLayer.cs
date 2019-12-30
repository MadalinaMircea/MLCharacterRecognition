using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [Serializable]
    class DropoutLayer : NetworkLayer
    {
        public double Rate { get; set; }

        [JsonIgnore]
        public LayerOutput Output { get; set; }

        [JsonConstructor]
        public DropoutLayer(double rate) : base("Dropout")
        {
            if (Rate > 0.9)
                throw new Exception("Rate must be between 0 and 1!");

            Rate = rate;
        }

        public override LayerOutput GetData()
        {
            return Output;
        }

        private void ComputeFilteredImage()
        {
            Random rnd = new Random();

            FilteredImage image = (FilteredImage)PreviousLayer.GetData();
            FilteredImageChannel[] newChannels = new FilteredImageChannel[image.NumberOfChannels];
            int size = image.Size;

            for (int i = 0; i < image.NumberOfChannels; i++)
            {
                double[,] newValues = new double[size, size];

                for (int valuesI = 0; valuesI < size; valuesI++)
                {
                    for (int valuesJ = 0; valuesJ < size; valuesJ++)
                    {
                        if (rnd.NextDouble() < Rate)
                        {
                            newValues[valuesI, valuesJ] = 0;
                        }
                        else
                        {
                            newValues[valuesI, valuesJ] = image.Channels[i].Values[valuesI, valuesJ];
                        }
                    }
                }

                newChannels[i] = new FilteredImageChannel(size, newValues);
            }

            Output = new FilteredImage(image.NumberOfChannels, newChannels);
        }

        private void ComputeFlattenedImage()
        {
            Random rnd = new Random();

            FlattenedImage previous = (FlattenedImage)PreviousLayer.GetData();

            double[] newValues = new double[previous.Size];

            for (int i = 0; i < previous.Size; i++)
            {
                if (rnd.NextDouble() < Rate)
                {
                    newValues[i] = 0;
                }
                else
                {
                    newValues[i] = previous.Values[i];
                }
            }

            Output = new FlattenedImage(previous.Size, newValues);
        }

        public override void ComputeOutput()
        {
            InitializeOutput();

            if (Output is FilteredImage)
            {
                ComputeFilteredImage();
            }
            else
            {
                ComputeFlattenedImage();
            }
        }

        private void InitializeOutput()
        {
            if (Output == null)
            {
                LayerOutput previousData = PreviousLayer.GetData();
                if (previousData is FlattenedImage)
                {
                    FlattenedImage previous = (FlattenedImage)previousData;
                    Output = new FlattenedImage(previous.Size);
                }
                else
                {
                    FilteredImage previous = (FilteredImage)previousData;
                    Output = new FilteredImage(previous.NumberOfChannels, previous.Size);
                }
            }
        }

        public override void CompileLayer(NetworkLayer previousLayer)
        {
            PreviousLayer = previousLayer;

            InitializeOutput();
        }
    }
}
