using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public class EvaluationMetrics
    {
        public double OverallAccuracy { get; set; }
        public int[,] ConfusionMatrix { get; set; }
        public int[] TP { get; set; }
        public int[] TN { get; set; }
        public int[] FN { get; set; }
        public int[] FP { get; set; }
        public double[] Sensitivity { get; set; }
        public double[] Specificity { get; set; }
        public double[] ClasswiseAccuracy { get; set; }
        public double[] ClasswiseMissclassification { get; set; }
        public double[] ClasswisePrecision { get; set; }

        public int Total { get; set; }
        public int ClassNr { get; set; }

        public EvaluationMetrics(int[,] matrix, int total)
        {
            ConfusionMatrix = matrix;
            Total = total;
            ClassNr = matrix.GetLength(0);
            ComputeMetrics();
        }

        [JsonConstructor]
        public EvaluationMetrics(double overallAccuracy, int[,] confusionMatrix,
            int[] tP, int[] tN, int[] fN, int[] fP, double[] sensitivity, double[] 
            specificity, double[] classwiseAccuracy,
            double[] classwiseMissclassification, double[] classwisePrecision, int total, int classNr)
        {
            OverallAccuracy = overallAccuracy;
            ConfusionMatrix = confusionMatrix;
            TP = tP;
            TN = tN;
            FN = fN;
            FP = fP;
            Sensitivity = sensitivity;
            Specificity = specificity;
            ClasswiseAccuracy = classwiseAccuracy;
            ClasswiseMissclassification = classwiseMissclassification;
            ClasswisePrecision = classwisePrecision;
            Total = total;
            ClassNr = classNr;
        }

        private void ComputeMetrics()
        {
            TP = new int[ClassNr];
            FN = new int[ClassNr];
            FP = new int[ClassNr];
            TN = new int[ClassNr];
            ClasswiseAccuracy = new double[ClassNr];
            Sensitivity = new double[ClassNr];
            Specificity = new double[ClassNr];
            ClasswiseMissclassification = new double[ClassNr];
            ClasswisePrecision = new double[ClassNr];

            int correctClassifications = 0;

            for (int classi = 0; classi < ClassNr; classi++)
            {
                //TP is on position [i,i], FN is the sum of the other columns
                //FP is on the column except for [i,i], TN is the sum of everything except the row and column of the class

                TP[classi] = ConfusionMatrix[classi, classi];

                for (int classj = 0; classj < ClassNr; classj++)
                {
                    if (classi != classj)
                    {
                        FN[classi] += ConfusionMatrix[classi, classj];
                        FP[classj] += ConfusionMatrix[classj, classi];
                    }
                }

                correctClassifications += TP[classi];
            }

            for (int classi = 0; classi < ClassNr; classi++)
            {
                TN[classi] = Total - TP[classi] - FN[classi] - FP[classi];
                ClasswiseAccuracy[classi] = (TP[classi] + TN[classi]) / (double)Total;
                ClasswisePrecision[classi] = (double)TP[classi] / (TP[classi] + FP[classi]);
                Sensitivity[classi] = (double)TP[classi] / (TP[classi] + FN[classi]);
                Specificity[classi] = (double)TN[classi] / (TN[classi] + FP[classi]);
                ClasswiseMissclassification[classi] = (double)(FP[classi] + FN[classi]) / Total;
            }

            OverallAccuracy = (double)correctClassifications / Total;
        }
    }
}
