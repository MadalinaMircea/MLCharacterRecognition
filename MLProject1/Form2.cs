using MLProject1.CNN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MLProject1
{
    public partial class Form2 : Form
    {
        Point lastPoint = Point.Empty;

        //used to see if the user is clicking or not
        bool isMouseDown = false;

        //KerasNeuralNetwork network;
        CNNController controller = new CNNController();

        int i = 0;

        public Form2()
        {
            InitializeComponent();
            //network = new KerasNeuralNetwork("smallModel.json", "smallWeights.h5", "smallBest.h5");
            //network = new ConvolutionalNeuralNetwork("newJson.json", "modelWeights5.h5");

            //string datasetPath = "data/new";
            //H5FileId fileId = H5F.open("tryToRead.h5", H5F.OpenMode.ACC_RDONLY);
            //H5DataSetId dataSetId = H5D.open(fileId, datasetPath);
            //H5DataTypeId typeId = H5D.getType(dataSetId);

            //// read array (shape may be inferred w/ H5S.get_simple_extent_ndims)
            //float[,] arr = new float[162, 128];
            //GCHandle gch = GCHandle.Alloc(arr, GCHandleType.Pinned);
            //try
            //{
            //    //H5D.read(dataSetId, typeId, gch.AddrOfPinnedObject());
            //}
            //finally
            //{
            //    gch.Free();
            //}

            GlobalRandom.InitializeRandom();

            Task[] tasks = new Task[3];

            tasks[0] = Task.Run(() =>
            {
                CNNController ctrl = new CNNController();
                ctrl.CreateAndCompileModel2();
                ctrl.WriteToFile("model1.json", "model1Weights");
                ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

                ctrl.Test(1);

                ctrl.Train(208, "model1trainedWeights", 1);
            });

            tasks[1] = Task.Run(() =>
            {
                CNNController ctrl = new CNNController();
                ctrl.CreateAndCompileModel3();
                ctrl.WriteToFile("model2.json", "model2Weights");
                ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

                ctrl.Test(2);

                ctrl.Train(416, "model2trainedWeights", 2);
            });

            tasks[2] = Task.Run(() =>
            {
                TotallyUselessGrayscaleController ctrl = new TotallyUselessGrayscaleController();
                ctrl.CreateAndCompileModel4();
                ctrl.WriteToFile("model3.json", "model3Weights");
                ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

                ctrl.Test(3);

                ctrl.Train(104, "model3trainedWeights", 3);
            });

            Task.WaitAll(tasks);

            //controller.CreateAndCompileModel("same.json", "sameWeights");
            ////controller.WriteToFile("same.json", "sameWeights");

            //controller.PrepareImageSets("data/Train", "data/Test", "data/Valid");

            //controller.Train(208, "trainedWeights");

            //controller.Test();
            ///

            //double[,] ch1 = new double[2, 2] { { 1, 2 }, { 3, 4 } };
            //double[,] ch2 = new double[2, 2] { { 5, 6 }, { 7, 8 } };


            //FilteredImageChannel c1 = new FilteredImageChannel(2, ch1);
            //FilteredImageChannel c2 = new FilteredImageChannel(2, ch2);

            //FilteredImageChannel[] chs = new FilteredImageChannel[2] { c1, c2 };

            //FilteredImage image = new FilteredImage(2, chs);

            //double[] output = new double[8];

            //int outputIndex = 0;

            //for (int channel = 0; channel < image.NumberOfChannels; channel++)
            //{
            //    for (int valuesI = 0; valuesI < image.Size; valuesI++)
            //    {
            //        for (int valuesJ = 0; valuesJ < image.Size; valuesJ++)
            //        {
            //            output[outputIndex] = image.Channels[channel].Values[valuesI, valuesJ];
            //            outputIndex++;
            //        }
            //    }

            //}

            //outputIndex = 0;

            //FlattenedImage img = new FlattenedImage(8, new double[8]{ 1, 2, 3, 4, 5, 6, 7, 8 });

            //FlattenedImage[] images = new FlattenedImage[2] { img, img };

            //FilteredImageChannel[] channels = new FilteredImageChannel[image.NumberOfChannels];

            //for (int channel = 0; channel < image.NumberOfChannels; channel++)
            //{
            //    double[,] values = new double[image.Size, image.Size];
            //    for (int valuesI = 0; valuesI < image.Size; valuesI++)
            //    {
            //        for (int valuesJ = 0; valuesJ < image.Size; valuesJ++)
            //        {
            //            values[valuesI, valuesJ] = ((FlattenedImage)images[outputIndex]).Values.Sum();
            //            outputIndex++;
            //        }
            //    }
            //    channels[channel] = new FilteredImageChannel(image.Size, values);
            //}



            //double[,] first = new double[5, 5];
            //for (int i = 0; i < 5; i++)
            //{
            //    for (int j = 0; j < 5; j++)
            //    {
            //        first[i, j] = i * 5 + j;
            //    }
            //}

            //double[,] second = new double[3, 3];
            //for (int i = 0; i < 3; i++)
            //{
            //    for (int j = 0; j < 3; j++)
            //    {
            //        second[i, j] = 9 - i * 3 - j;
            //    }
            //}

            //double[,] result = MatrixUtils.FullConvolution(first, second);


            //EvaluationMetrics metrics = controller.Evaluate();
            //Console.WriteLine("Accuracy: " + metrics.Accuracy);
            //Console.WriteLine("Error: " + metrics.Error);

            //H5FileId fileId = H5F.open("tryToRead.h5", H5F.OpenMode.ACC_RDONLY);

            //H5GroupId groupId = H5G.open(fileId, "/model_weights/conv2d_1/conv2d_1");

            //H5DataSetId datasetId = H5D.open(groupId, "bias:0");

            //H5DataTypeId datatypeId = H5D.getType(datasetId);

            //float[,] arr = new float[32,1];
            //H5Array<float> array = new H5Array<float>(arr);

            //H5D.read<float>(datasetId, datatypeId, array);





            ClearPicture();

        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            //assign to lastPoint the current mouse position
            lastPoint = e.Location;

            //user is currently clicking
            isMouseDown = true;
        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            //if the mouse button is down
            if (isMouseDown == true)
            {
                //if our last point is not null
                if (lastPoint != null)
                {
                    Graphics g = Graphics.FromImage(pictureBox1.Image);
                    g.DrawEllipse(new Pen(Color.Black, 5), new RectangleF(e.Location, new SizeF(5,5)));
                    g.SmoothingMode = SmoothingMode.HighQuality;

                    //refresh the picturebox
                    pictureBox1.Invalidate();

                    //assign the location to lastPoint
                    lastPoint = e.Location;
                }
            }
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            //user is no longer clicking
            isMouseDown = false;

            //the last clicked point is reinitialized
            lastPoint = Point.Empty;
        }

        private void clearButton_Click(object sender, EventArgs e)
        {
            ClearPicture();
        }

        private void ClearPicture()
        {
            //assign a new bitmap to the picturebox.Image property
            pictureBox1.Image = ImageProcessing.CreateInitialImage(pictureBox1.Width, pictureBox1.Height); 

            //refresh the pictureBox
            Invalidate();
        }

        private void RecogniseButton_Click(object sender, EventArgs e)
        {
            //string path = Environment.CurrentDirectory + "\\image.jpg";

            Bitmap img = ImageProcessing.CropWhite(pictureBox1.Image, 75, 75);

            predictionLabel.Text = controller.RecogniseImage(img).ToString();


            ClearPicture();
        }

        private void SaveButton_Click(object sender, EventArgs e)
        {
            //string path = Environment.CurrentDirectory + "\\image.jpg";
            ImageProcessing.SaveImage(ImageProcessing.ResizeImage(pictureBox1.Image, 100, 75), 
                Environment.CurrentDirectory + "\\data\\image" + i + ".jpg");
            i++;
            ClearPicture();
        }

        private void ImportButton_Click(object sender, EventArgs e)
        {
            //DialogResult result = openFileDialog1.ShowDialog();
            //if (result == DialogResult.OK)
            //{
            //    string file = openFileDialog1.FileName;
            //    try
            //    {
            //        predictionLabel.Text = controller.RecogniseImage(new Bitmap(file)).ToString();

            //        Bitmap bmp = new Bitmap(file);
            //        pictureBox1.Image = bmp;
            //        pictureBox1.Invalidate();
            //    }
            //    catch (IOException)
            //    {
            //    }
            //}
        }

        private void CropButton_Click(object sender, EventArgs e)
        {
            //ImageProcessing.CropWhite(pictureBox1.Image, Environment.CurrentDirectory +, 100, 75);
        }
    }
}
