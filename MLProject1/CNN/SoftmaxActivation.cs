using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    class SoftmaxActivation : Activation
    {
        public override string ToString()
        {
            return "softmax";
        }

        public override FlattenedImage Activate(FlattenedImage img)
        {
            double sum = 0;

            double[] result = new double[img.Size];

            for(int i = 0; i < img.Size; i++)
            {
                result[i] = Math.Exp(img.Values[i]);
                sum += result[i];
            }

            for(int i = 0; i < img.Size; i++)
            {
                result[i] /= sum;
            }

            return new FlattenedImage(img.Size, result);
        }

        public override FlattenedImage GetDerivative(FlattenedImage image, int correctClass)
        {
            //FlattenedImage image = (FlattenedImage)output;

            //double[] result = new double[image.Size];

            //double totalSum = 0;

            //for(int i = 0; i < image.Size; i++)
            //{
            //    totalSum += Math.Exp(image.Values[i]);
            //}

            //for(int i = 0; i < image.Size; i++)
            //{
            //    double e = Math.Exp(image.Values[i]);
            //    result[i] = (e * (totalSum - e)) / (totalSum * totalSum);
            //}

            //return new FlattenedImage(image.Size, result);


            double[] result = new double[image.Size];

            double totalSum = 0;

            for (int i = 0; i < image.Size; i++)
            {
                result[i] = Math.Exp(image.Values[i]);
                totalSum += result[i];
            }

            double squareSum = totalSum * totalSum;

            for (int i = 0; i < image.Size; i++)
            {
                if (i == correctClass)
                {
                    result[i] = (result[i] * (totalSum - result[i])) / squareSum;
                }
                else
                {
                    result[i] = (-result[correctClass] * result[i]) / squareSum;
                }
            }

            return new FlattenedImage(image.Size, result);
        }
    }
}
