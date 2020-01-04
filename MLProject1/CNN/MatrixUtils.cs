using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    static class MatrixUtils
    {
        static void ReverseColumns(double[,] matrix)
        {
            double aux;
            for (int i = 0; i < matrix.GetLength(1); i++)
            {
                int k = matrix.GetLength(1) - 1;
                for (int j = 0;j < k; j++)
                {
                    aux = matrix[j, i];
                    matrix[j, i] = matrix[k, i];
                    matrix[k, i] = aux;
                    k--;
                }
            }
        }

        public static double[,] Convolve(double[,] matrix, double[,] kernel)
        {
            int resultSize = matrix.GetLength(0) - kernel.GetLength(0) + 1;
            double[,] result = new double[resultSize, resultSize];

            Task[] tasks = new Task[resultSize];

            for (int outputI = 0; outputI < resultSize; outputI++)
            {
                int taski = 0 + outputI;

                tasks[taski] = Task.Run(() =>
                {
                    for (int outputJ = 0; outputJ < resultSize ; outputJ++)
                    {
                        double total = 0;
                        for (int kernelI = 0; kernelI < kernel.GetLength(0); kernelI++)
                        {
                            for (int kernelJ = 0; kernelJ < kernel.GetLength(0); kernelJ++)
                            {
                                int indexI = taski + kernelI;
                                int indexJ = outputJ + kernelJ;
                                total += matrix[indexI, indexJ] * kernel[kernelI, kernelJ];
                            }
                        }
                        result[taski, outputJ] = total;
                    }
                });
            }

            Task.WaitAll(tasks);

            //for (int outputI = 0; outputI < resultSize; outputI++)
            //{
            //    for (int outputJ = 0; outputJ < resultSize; outputJ++)
            //    {
            //        double total = 0;
            //        for (int kernelI = 0; kernelI < kernel.GetLength(0); kernelI++)
            //        {
            //            for (int kernelJ = 0; kernelJ < kernel.GetLength(0); kernelJ++)
            //            {
            //                int indexI = outputI + kernelI;
            //                int indexJ = outputJ + kernelJ;
            //                total += matrix[indexI, indexJ] * kernel[kernelI, kernelJ];
            //            }
            //        }
            //        result[outputI, outputJ] = total;
            //    }
            //}

            return result;
        }

        public static double[,] ConvolveSame(double[,] matrix, double[,] kernel)
        {
            double[,] result = new double[matrix.GetLength(0), matrix.GetLength(0)];
            int halfSize = kernel.GetLength(0) / 2;

            Task[] tasks = new Task[matrix.GetLength(0)];

            for (int outputI = 0; outputI < matrix.GetLength(0); outputI++)
            {
                int taski = 0 + outputI;

                tasks[taski] = Task.Run(() =>
                {
                    for (int outputJ = 0; outputJ < matrix.GetLength(0); outputJ++)
                    {
                        double total = 0;
                        for (int kernelI = 0; kernelI < kernel.GetLength(0); kernelI++)
                        {
                            for (int kernelJ = 0; kernelJ < kernel.GetLength(0); kernelJ++)
                            {
                                int indexI = taski + kernelI - halfSize;
                                int indexJ = outputJ + kernelJ - halfSize;

                                if (indexI >= 0 && indexI < matrix.GetLength(0) && indexJ >= 0 && indexJ < matrix.GetLength(0))
                                {
                                    total += matrix[indexI, indexJ] * kernel[kernelI, kernelJ];
                                }
                            }
                        }
                        result[taski, outputJ] = total;
                    }
                });
            }

            Task.WaitAll(tasks);

            //for (int outputI = 0; outputI < matrix.GetLength(0); outputI++)
            //{
            //    for (int outputJ = 0; outputJ < matrix.GetLength(0); outputJ++)
            //    {
            //        double total = 0;
            //        for (int kernelI = 0; kernelI < kernel.GetLength(0); kernelI++)
            //        {
            //            for (int kernelJ = 0; kernelJ < kernel.GetLength(0); kernelJ++)
            //            {
            //                int indexI = outputI + kernelI - halfSize;
            //                int indexJ = outputJ + kernelJ - halfSize;

            //                if (indexI >= 0 && indexI < matrix.GetLength(0) && indexJ >= 0 && indexJ < matrix.GetLength(0))
            //                {
            //                    total += matrix[indexI, indexJ] * kernel[kernelI, kernelJ];
            //                }
            //            }
            //        }
            //        result[outputI, outputJ] = total;
            //    }
            //}

            return result;
        }

        public static double[,] FullConvolution(double[,] matrix, double[,] kernel)
        {
            int kernelSize = kernel.GetLength(0);
            int matrixSize = matrix.GetLength(0);
            int outputSize = matrix.GetLength(0) + kernelSize - 1;
            double[,] result = new double[outputSize, outputSize];

            Task[] tasks = new Task[outputSize];

            for (int outputI = 0; outputI < outputSize; outputI++)
            {
                int taski = 0 + outputI;

                tasks[taski] = Task.Run(() =>
                    {
                        for (int outputJ = 0; outputJ < outputSize; outputJ++)
                        {
                            double total = 0;
                            for (int kernelI = 0; kernelI < kernelSize; kernelI++)
                            {
                                for (int kernelJ = 0; kernelJ < kernelSize; kernelJ++)
                                {
                                    int indexI = taski + kernelI - kernelSize + 1;
                                    int indexJ = outputJ + kernelJ - kernelSize + 1;

                                    if (indexI >= 0 && indexI < matrixSize && indexJ >= 0 && indexJ < matrixSize)
                                    {
                                        total += matrix[indexI, indexJ] * kernel[kernelI, kernelJ];
                                    }
                                }
                            }
                            result[taski, outputJ] = total;
                        }
                    });
            }

            Task.WaitAll(tasks);

            return result;
        }

        public static double[,] FullConvolutionSame(double[,] matrix, double[,] kernel)
        {
            int kernelSize = kernel.GetLength(0);
            int matrixSize = matrix.GetLength(0);
            int outputSize = matrix.GetLength(0) + kernelSize - 1;
            double[,] result = new double[outputSize, outputSize];

            Task[] tasks = new Task[outputSize];

            for (int outputI = 0; outputI < outputSize; outputI++)
            {
                int taski = 0 + outputI;

                tasks[taski] = Task.Run(() =>
                {
                    for (int outputJ = 0; outputJ < outputSize; outputJ++)
                    {
                        double total = 0;
                        for (int kernelI = 0; kernelI < kernelSize; kernelI++)
                        {
                            for (int kernelJ = 0; kernelJ < kernelSize; kernelJ++)
                            {
                                int indexI = taski + kernelI - kernelSize + 1;
                                int indexJ = outputJ + kernelJ - kernelSize + 1;

                                if (indexI >= 0 && indexI < matrixSize && indexJ >= 0 && indexJ < matrixSize)
                                {
                                    total += matrix[indexI, indexJ] * kernel[kernelI, kernelJ];
                                }
                            }
                        }
                        result[taski, outputJ] = total;
                    }
                });
            }

            Task.WaitAll(tasks);

            double[,] centralResult = new double[kernelSize, kernelSize];

            int halfSize = matrixSize / 2;

            for (int i = halfSize; i < kernelSize + halfSize; i++)
            {
                for (int j = halfSize; j < kernelSize + halfSize; j++)
                {
                    centralResult[i - halfSize, j - halfSize] = result[i, j];
                }
            }

            return centralResult;
        }

        static void Transpose(double[,] matrix)
        {
            double aux;

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = i; j < matrix.GetLength(1); j++)
                {
                    aux = matrix[i, j];
                    matrix[i, j] = matrix[j, i];
                    matrix[j, i] = aux;
                }
            }
        }
        public static double[,] Rotate180(double[,] matrix)
        {
            Transpose(matrix);
            ReverseColumns(matrix);
            Transpose(matrix);
            ReverseColumns(matrix);

            return matrix;
        }

        public static double ElementSum(double[,] matrix)
        {
            double total = 0;

            for(int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    total += matrix[i, j];
                }
            }

            return total;
        }
    }
}
