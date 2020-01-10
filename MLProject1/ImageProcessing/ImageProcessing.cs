using MLProject1.CNN;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1
{
    public class ImageProcessing
    {
        public static Bitmap CreateInitialImage(int width, int height)
        {
            //a new Bitmap image is created
            Bitmap bmp = new Bitmap(width, height);
            bmp.SetResolution(96f, 96f);
            Graphics g = Graphics.FromImage(bmp);
            g.FillRectangle(new SolidBrush(Color.White), 0, 0, width, height);

            return bmp;
        }
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        private static Image ResizeImageWithRatio(Image image, int canvasWidth, int canvasHeight,
                     int originalWidth, int originalHeight)
        {
            System.Drawing.Image thumbnail = new Bitmap(canvasWidth, canvasHeight); // changed parm names
            System.Drawing.Graphics graphic =
                         System.Drawing.Graphics.FromImage(thumbnail);

            graphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphic.SmoothingMode = SmoothingMode.HighQuality;
            graphic.PixelOffsetMode = PixelOffsetMode.HighQuality;
            graphic.CompositingQuality = CompositingQuality.HighQuality;

            /* ------------------ new code --------------- */

            // Figure out the ratio
            double ratioX = (double)canvasWidth / (double)originalWidth;
            double ratioY = (double)canvasHeight / (double)originalHeight;
            // use whichever multiplier is smaller
            double ratio = ratioX < ratioY ? ratioX : ratioY;

            // now we can get the new height and width
            int newHeight = Convert.ToInt32(originalHeight * ratio);
            int newWidth = Convert.ToInt32(originalWidth * ratio);

            // Now calculate the X,Y position of the upper-left corner 
            // (one of these will always be zero)
            int posX = Convert.ToInt32((canvasWidth - (originalWidth * ratio)) / 2);
            int posY = Convert.ToInt32((canvasHeight - (originalHeight * ratio)) / 2);

            graphic.Clear(Color.White); // white padding
            graphic.DrawImage(image, posX, posY, newWidth, newHeight);

            /* ------------- end new code ---------------- */

            System.Drawing.Imaging.ImageCodecInfo[] info =
                             ImageCodecInfo.GetImageEncoders();
            EncoderParameters encoderParameters;
            encoderParameters = new EncoderParameters(1);
            encoderParameters.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality,
                             100L);

            return thumbnail;
        }

        public static void SaveImage(Image image, string path)
        {
            image.Save(path, ImageFormat.Jpeg);
        }

        private static bool IsWhite(Color pixel)
        {
            return (pixel.R == 255 && pixel.G == 255 && pixel.B == 255);
        }

        internal static FlattenedImage GetNormalizedFlattenedImage(Bitmap bitmap)
        {
            using (Bitmap bmp = new Bitmap(bitmap))
            {
                int height = bitmap.Height;

                int size = height * height * 3;

                double[] values = new double[size];
                int index = 0;

                for (int i = 0; i < bitmap.Height; i++)
                {
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        Color color = bmp.GetPixel(j, i);
                        //values[index] = (255 - color.R) / 255.0;
                        //index++;
                        //values[index] = (255 - color.G) / 255.0;
                        //index++;
                        //values[index] = (255 - color.B) / 255.0;
                        //index++;

                        values[index] = color.R / 255.0;
                        index++;
                        values[index] = color.G / 255.0;
                        index++;
                        values[index] = color.B / 255.0;
                        index++;

                    }
                }


                FlattenedImage image = new FlattenedImage(size, values);

                return image;
            }
        }

        internal static FlattenedImage GetNormalizedGrayscaleFlattenedImage(Bitmap bitmap)
        {
            using (Bitmap bmp = new Bitmap(bitmap))
            {
                int height = bitmap.Height;

                int size = height * height;

                double[] values = new double[size];
                int index = 0;

                for (int i = 0; i < bitmap.Height; i++)
                {
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        Color color = bmp.GetPixel(j, i);
                        //values[index] = (255 - (color.R + color.G + color.B) / 3) / 255.0;
                        values[index] = ((color.R + color.G + color.B) / 3) / 255.0;
                        index++;
                    }
                }


                FlattenedImage image = new FlattenedImage(size, values);

                return image;
            }
        }

        private static bool IsRowWhite(Bitmap image, int row)
        {
            for (int i = 0; i < image.Width; i++)
            {
                if (!IsWhite(image.GetPixel(i, row)))
                    return false;
            }
            return true;
        }

        private static bool IsColumnWhite(Bitmap image, int column)
        {
            for (int j = 0; j < image.Height; j++)
            {
                if (!IsWhite(image.GetPixel(column, j)))
                    return false;
            }
            return true;
        }

        private static List<int> GetWhiteLimits(Image image)
        {
            Bitmap bmp = new Bitmap(image);

            int topRow = 0, bottomRow = image.Height - 1, leftCol = 0, rightCol = image.Width - 1;

            for (int i = 0; i < image.Height; i++)
            {
                if (IsRowWhite(bmp, i))
                {
                    topRow = i;
                }
                else
                {
                    break;
                }
            }

            for (int i = image.Height - 1; i >= 0; i--)
            {
                if (IsRowWhite(bmp, i))
                {
                    bottomRow = i;
                }
                else
                {
                    break;
                }
            }

            for (int i = 0; i < image.Width; i++)
            {
                if (IsColumnWhite(bmp, i))
                {
                    leftCol = i;
                }
                else
                {
                    break;
                }
            }

            for (int i = image.Width - 1; i >= 0; i--)
            {
                if (IsColumnWhite(bmp, i))
                {
                    rightCol = i;
                }
                else
                {
                    break;
                }
            }

            List<int> result = new List<int>();
            result.Add(topRow);
            result.Add(bottomRow);
            result.Add(leftCol);
            result.Add(rightCol);

            return result;
        }

        public static Bitmap CropWhite(Image image, int width, int height)
        {
            //top row, bottom row, left column, right column
            List<int> limits = GetWhiteLimits(image);

            int newHeight = limits[1] - limits[0];
            int newWidth = limits[3] - limits[2];

            Image newImage = new Bitmap(newWidth, newHeight);
            using (Graphics g = Graphics.FromImage(newImage))
            {
                g.DrawImage(image,
                  new RectangleF(0, 0, newWidth, newHeight),
                  new RectangleF(limits[2], limits[0], newWidth, newHeight),
                  GraphicsUnit.Pixel);
            }

            Image resized = ResizeImageWithRatio(newImage, width, height, newWidth, newHeight);

            //Bitmap resized = ResizeImage(newImage, width, height);

            //if (newHeight < newWidth)
            //{
            //    resized = ResizeImage(newImage, width, (int)(((double)width / (double)newWidth) * height));

            //}
            //else
            //{
            //    resized = ResizeImage(newImage, (int)(((double)height / (double)newHeight) * image.Width), height);
            //}

            return new Bitmap(resized);
        }

        public static void RemoveWhiteDataset()
        {
            int i = 0;
            foreach (string type in new List<string>() { "Alphabet Training", "Alphabet Testing", "Alphabet Validation" })
            {
                Directory.CreateDirectory("new\\Square\\" + type);

                foreach (char c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                {
                    Directory.CreateDirectory("new\\Square\\" + type + "\\" + c);

                    string[] files = Directory.GetFiles(Environment.CurrentDirectory + "\\data\\" + type + "\\" + c);

                    foreach (string file in files)
                    {
                        using (Image img = Image.FromFile(file))
                        {
                            SaveImage(ImageProcessing.CropWhite(img, 75, 75), "new\\Square\\" + type + "\\" + c + "\\image" + i + ".jpg");
                            i++;
                        }
                    }
                }
            }
        }

        //private static RGBPixel[,] GetRGBMatrix(Bitmap image)
        //{
        //    int width = image.Width, height = image.Height;

        //    RGBPixel[,] result = new RGBPixel[height, width];

        //    for (int i = 0; i < height; i++)
        //    {
        //        for (int j = 0; j < width; j++)
        //        {
        //            Color color = image.GetPixel(j, i);
        //            result[i, j] = new RGBPixel(color.R, color.G, color.B);
        //        }
        //    }

        //    return result;
        //}

        //private static Bitmap ReconstructFromRGBMatrix(RGBPixel[,] imageMatrix)
        //{
        //    int width = imageMatrix.GetLength(1), height = imageMatrix.GetLength(0);

        //    Bitmap image = new Bitmap(width, height);
        //    using (Graphics graph = Graphics.FromImage(image))
        //    {
        //        Rectangle rectangle = new Rectangle(0, 0, width, height);
        //        graph.FillRectangle(Brushes.White, rectangle);
        //    }

        //    for (int i = 0; i < height; i++)
        //    {
        //        for (int j = 0; j < width; j++)
        //        {
        //            image.SetPixel(j, i, Color.FromArgb((int)imageMatrix[i, j].Red, (int)imageMatrix[i, j].Green, (int)imageMatrix[i, j].Blue));
        //        }
        //    }

        //    return image;
        //}

        //public static Bitmap ReconstructFromNormalizedRGBMatrix(RGBPixel[,] imageMatrix)
        //{
        //    return ReconstructFromRGBMatrix(DenormalizeRGBMatrix(imageMatrix));
        //}

        //private static RGBPixel[,] DenormalizeRGBMatrix(RGBPixel[,] imageMatrix)
        //{
        //    for (int i = 0; i < imageMatrix.GetLength(0); i++)
        //    {
        //        for (int j = 0; j < imageMatrix.GetLength(1); j++)
        //        {
        //            imageMatrix[i, j] = DenormalizePixel(imageMatrix[i, j]);
        //        }
        //    }

        //    return imageMatrix;
        //}

        //private static double DenormalizeValue(double value)
        //{
        //    return (int)(value * 255);
        //}

        //private static RGBPixel DenormalizePixel(RGBPixel pixel)
        //{
        //    return new RGBPixel(DenormalizeValue(pixel.Red), DenormalizeValue(pixel.Green), DenormalizeValue(pixel.Blue));
        //}

        //private static double NormalizeValue(double value)
        //{
        //    return value / 255.0;
        //}

        //private static RGBPixel NormalizePixel(RGBPixel pixel)
        //{
        //    return new RGBPixel(NormalizeValue(pixel.Red), NormalizeValue(pixel.Green), NormalizeValue(pixel.Blue));
        //}

        //private static RGBPixel[,] NormalizeRGBMatrix(RGBPixel[,] imageMatrix)
        //{
        //    for(int i = 0; i < imageMatrix.GetLength(0); i++)
        //    {
        //        for(int j = 0; j < imageMatrix.GetLength(1); j++)
        //        {
        //            imageMatrix[i, j] = NormalizePixel(imageMatrix[i,j]);
        //        }
        //    }

        //    return imageMatrix;
        //}

        //public static RGBPixel[,] GetNormalizedRGBMatrix(string path)
        //{
        //    using (Bitmap bmp = new Bitmap(path))
        //    {
        //        return NormalizeRGBMatrix(GetRGBMatrix(bmp));
        //    }
        //}

        //public static RGBPixel[,] GetNormalizedRGBMatrix(Bitmap bitmap)
        //{
        //    using (Bitmap bmp = new Bitmap(bitmap))
        //    {
        //        return NormalizeRGBMatrix(GetRGBMatrix(bmp));
        //    }
        //}

        public static FilteredImage GetNormalizedFilteredImage(Bitmap bitmap)
        {
            using (Bitmap bmp = new Bitmap(bitmap))
            {
                int height = bitmap.Height;

                FilteredImageChannel[] channels = new FilteredImageChannel[3];

                double[,] redChannel = new double[height, height];
                double[,] greenChannel = new double[height, height];
                double[,] blueChannel = new double[height, height];


                for (int i = 0; i < bitmap.Height; i++)
                {
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        Color color = bitmap.GetPixel(j, i);
                        redChannel[i, j] = (255 - color.R) / 255.0 - 0.5;
                        greenChannel[i, j] = (255 - color.G) / 255.0 - 0.5;
                        blueChannel[i, j] = (255 - color.B) / 255.0 - 0.5;

                        //redChannel[i, j] = (255 - color.R) / 255.0;
                        //greenChannel[i, j] = (255 - color.G) / 255.0;
                        //blueChannel[i, j] = (255 - color.B) / 255.0;
                    }
                }

                channels[0] = new FilteredImageChannel(height, redChannel);
                channels[1] = new FilteredImageChannel(height, greenChannel);
                channels[2] = new FilteredImageChannel(height, blueChannel);


                FilteredImage image = new FilteredImage(3, channels);

                return image;
            }
        }

        public static FilteredImage GetNormalizedMnist(Bitmap bitmap)
        {
            using (Bitmap bmp = new Bitmap(bitmap))
            {
                int height = bitmap.Height;

                FilteredImageChannel[] channels = new FilteredImageChannel[1];

                double[,] channel = new double[height, height];


                for (int i = 0; i < bitmap.Height; i++)
                {
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        Color color = bitmap.GetPixel(j, i);
                        channel[i, j] = ((color.R + color.G + color.B) / 3) / 255.0 - 0.5;
                    }
                }

                channels[0] = new FilteredImageChannel(height, channel);


                FilteredImage image = new FilteredImage(1, channels);

                return image;
            }
        }

        public static FilteredImage GetNormalizedGrayscaleFilteredImage(Bitmap bitmap)
        {
            using (Bitmap bmp = new Bitmap(bitmap))
            {
                int height = bitmap.Height;

                FilteredImageChannel[] channels = new FilteredImageChannel[1];

                double[,] channel = new double[height, height];


                for (int i = 0; i < bitmap.Height; i++)
                {
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        Color color = bitmap.GetPixel(j, i);
                        //redChannel[i, j] = color.R / 255.0;
                        //greenChannel[i, j] = color.G / 255.0;
                        //blueChannel[i, j] = color.B / 255.0;

                        channel[i, j] = (255 - ((color.R + color.G + color.B) / 3)) / 255.0 - 0.5;
                    }
                }

                channels[0] = new FilteredImageChannel(height, channel);


                FilteredImage image = new FilteredImage(1, channels);

                return image;
            }
        }
    }
}
