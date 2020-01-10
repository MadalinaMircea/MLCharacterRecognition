using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    [JsonConverter(typeof(ToStringJsonConverter))]
    public class SoftmaxActivation : Activation
    {
        FlattenedImage lastOutput;
        public override string ToString()
        {
            return "softmax";
        }

        private double[] CopyArray(double[] arr)
        {
            int size = arr.Length;
            double[] result = new double[size];

            for (int i = 0; i < size; i++)
            {
                result[i] = arr[i];
            }

            return result;
        }

        public override FlattenedImage Activate(FlattenedImage img)
        {
            lastOutput = new FlattenedImage(img.Size, CopyArray(img.Values));

            double sum = 0;

            double[] result = new double[img.Size];

            double maxx = img.Values.Max();

            for (int i = 0; i < img.Size; i++)
            {
                result[i] = Math.Exp(img.Values[i]);// - maxx);
                sum += result[i];
            }

            for(int i = 0; i < img.Size; i++)
            {
                result[i] /= sum;
            }

            return new FlattenedImage(img.Size, result);
        }

        public override FlattenedImage GetDerivative(FlattenedImage gradient, int correctClass)
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

            double correctClassGradient = gradient.Values[0];

            double[] result = new double[lastOutput.Size];

            double totalSum = 0;

            double maxx = lastOutput.Values.Max();

            for (int i = 0; i < lastOutput.Size; i++)
            {
                result[i] = Math.Exp(lastOutput.Values[i]);// - maxx);
                totalSum += result[i];
            }

            double squareSum = totalSum * totalSum;

            for (int i = 0; i < result.Length; i++)
            {
                if (i == correctClass)
                {
                    result[i] = correctClassGradient * ((result[i] * (totalSum - result[i])) / squareSum);
                }
                else
                {
                    result[i] = correctClassGradient * ((-result[correctClass] * result[i]) / squareSum);
                }
            }

            return new FlattenedImage(result.Length, result);
        }
    }
}
