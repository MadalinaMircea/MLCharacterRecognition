using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public abstract class PiecewiseActivation : Activation
    {
        public override FlattenedImage Activate(FlattenedImage img)
        {
            double[] result = new double[img.Size];

            for (int i = 0; i < img.Size; i++)
            {
                result[i] = ActivateValue(img.Values[i]);
            }

            return new FlattenedImage(img.Size, result);
        }

        public override FilteredImage Activate(FilteredImage img)
        {
            FilteredImageChannel[] resultChannels = new FilteredImageChannel[img.NumberOfChannels];
            double[,] resultValues;

            for (int c = 0; c < img.NumberOfChannels; c++)
            {
                resultValues = new double[img.Size, img.Size];

                for (int i = 0; i < img.Size; i++)
                {
                    for (int j = 0; j < img.Size; j++)
                    {
                        resultValues[i, j] = ActivateValue(img.Channels[c].Values[i, j]);
                    }
                }

                resultChannels[c] = new FilteredImageChannel(img.Size, resultValues);
            }

            return new FilteredImage(img.NumberOfChannels, resultChannels);
        }

        public override FlattenedImage GetDerivative(FlattenedImage output)
        {
            double[] result = new double[output.Size];

            for (int i = 0; i < output.Size; i++)
            {
                result[i] = GetValueDerivative(output.Values[i]);
            }

            return new FlattenedImage(output.Size, result);
        }

        public override FilteredImage GetDerivative(FilteredImage output)
        {
            FilteredImageChannel[] resultChannels = new FilteredImageChannel[output.NumberOfChannels];
            double[,] resultValues;

            for (int c = 0; c < output.NumberOfChannels; c++)
            {
                resultValues = new double[output.Size, output.Size];

                for (int i = 0; i < output.Size; i++)
                {
                    for (int j = 0; j < output.Size; j++)
                    {
                        resultValues[i, j] = GetValueDerivative(output.Channels[c].Values[i, j]);
                    }
                }

                resultChannels[c] = new FilteredImageChannel(output.Size, resultValues);
            }

            return new FilteredImage(output.NumberOfChannels, resultChannels);
        }

        public abstract double ActivateValue(double value);
        public abstract double GetValueDerivative(double value);
    }
}
