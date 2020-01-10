using Keras.Datasets;
using MLProject1.CNN;
using Newtonsoft.Json;
using Numpy;
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
using System.Threading;
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

        string path = Environment.GetFolderPath(Environment.SpecialFolder.Desktop) + "\\UnityCNN";
        string imagePath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop) + "\\UnityCNN\\image.jpg";
        string newImagePath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop) + "\\UnityCNN\\image2.jpg";
        string responsePath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop) + "\\UnityCNN\\image.txt";

        public char GetPrediction(KerasController ctrl)
        {
            using (Bitmap img = new Bitmap(imagePath))
            {
                Bitmap bmp = ImageProcessing.CropWhite(img, 75, 75);
                bmp.RotateFlip(RotateFlipType.Rotate180FlipNone);
                bmp.Save(newImagePath);

                img.Dispose();

                return ctrl.RecogniseImage(newImagePath);
            }
        }

        public char GetPrediction(CNNController ctrl)
        {
            using (Bitmap img = new Bitmap(imagePath))
            {
                Bitmap bmp = ImageProcessing.CropWhite(img, 75, 75);
                bmp.RotateFlip(RotateFlipType.Rotate180FlipNone);

                img.Dispose();

                return ctrl.RecogniseImage(bmp);
            }
        }

        public Form2()
        {
            InitializeComponent();

            if(!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }

            //StartCNNForUnity();

            //StartKerasForUnity();

            //StartCNNForForm();

            //StartTrainingMnist();

            //StartTrainingTasks();

            //StartTrainingCNN();

            //EvaluateCNN();

            ClearPicture();

            //StartCNNForForm();

            ConfusionMatrixForm form = new ConfusionMatrixForm();
            form.Show();

        }

        private void StartCNNForForm()
        {
            //controller.CreateAndCompileModel3();
            //controller.WriteToFile("modelCorrectFinal.json", "modelCorrectFinal");

            //controller.PrepareImageSets("data/Train", "data/Test", "data/Valid");

            //controller.Train(408, "modelCorrectFinal", 1, 0.1);

            controller.CreateAndCompileModel("model6final.json", "model6final2");
        }

        private void StartTrainingTasks()
        {
            Task[] tasks = new Task[2];

            tasks[0] = Task.Run(() =>
            {
                try
                {
                    CNNController ctrl = new CNNController();
                    ctrl.CreateAndCompileModel5();
                    ctrl.WriteToFile("model5final.json", "model5final");
                    ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

                    //ctrl.Test(1);

                    ctrl.Train(208, "model5final", 1, 0.05);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Process 1: " + ex.Message);
                }
            });

            tasks[1] = Task.Run(() =>
            {
                try
                {
                    CNNController ctrl = new CNNController();
                    ctrl.CreateAndCompileModel6();
                    ctrl.WriteToFile("model6final.json", "model6final");
                    ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

                    ctrl.Test(2);

                    ctrl.Train(416, "model6final", 2, 0.005);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Process 2: " + ex.ToString());
                }
            });

            Task.WaitAll(tasks);
        }

        private void StartKerasForUnity()
        {
            KerasController ctrl = new KerasController("smallModelBest.json", "smallBest.h5");

            while (true)
            {
                if (File.Exists(imagePath))
                {
                    try
                    {
                        Thread.Sleep(500);
                        char c = GetPrediction(ctrl);
                        File.WriteAllText(responsePath, c.ToString());
                        Console.WriteLine("Predicted: " + c);
                        File.Delete(newImagePath);
                        File.Delete(imagePath);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                }
            }
        }

        private void StartCNNForUnity()
        {
            controller.CreateAndCompileModel("model6final.json", "model6final2");
            Console.WriteLine("Ready");

            while (true)
            {
                if (File.Exists(imagePath))
                {
                    try
                    {
                        Thread.Sleep(500);
                        char c = GetPrediction(controller);
                        File.WriteAllText(responsePath, c.ToString());
                        Console.WriteLine("Predicted: " + c);
                        File.Delete(newImagePath);
                        File.Delete(imagePath);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                }
            }
        }

        private void StartTrainingCNN()
        {
            CNNController ctrl = new CNNController();

            ctrl.CreateAndCompileModel("model6final.json", "model6final2");

            //ctrl.WriteToFile();
            ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

            //ctrl.Test(2);

            while (true)
            {
                try
                {
                    ctrl.Train(416, "model6final2", 2, 0.000025);
                }
                catch(Exception e)
                {
                    ctrl.CreateAndCompileModel("model6final.json", "model6final2");
                }

                ctrl.ShuffleSets();
            }

        }

        private void EvaluateCNN()
        {
            CNNController ctrl = new CNNController();

            ctrl.CreateAndCompileModel("model6final.json", "model6final2");

            ctrl.PrepareImageSets("data/Train", "data/Test", "data/Valid");

            ctrl.EvaluateModel();
        }

        private void StartTrainingMnist()
        {
            controller.CreateAndCompileModel("mnist.json", "mnist");

            controller.PrepareImageSets("E:\\Programs\\MNIST-JPG-master\\output\\training", 
                "E:\\Programs\\MNIST-JPG-master\\output\\testing", "E:\\Programs\\MNIST-JPG-master\\output\\testing");

            controller.TrainOneMnist(0.005);

            //controller.Train(52, "trainedWeights", 1, 0.005);

            //controller.Test(1);

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
            Bitmap img = ImageProcessing.CropWhite(pictureBox1.Image, 75, 75);

            predictionLabel.Text = controller.RecogniseImage(img).ToString();


            ClearPicture();
        }

        private void SaveButton_Click(object sender, EventArgs e)
        {
            ImageProcessing.SaveImage(ImageProcessing.ResizeImage(pictureBox1.Image, 100, 75), 
                Environment.CurrentDirectory + "\\data\\image" + i + ".jpg");
            i++;
            ClearPicture();
        }

        private void ImportButton_Click(object sender, EventArgs e)
        {
            DialogResult result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
            {
                string file = openFileDialog1.FileName;
                try
                {
                    using (Bitmap bmp = new Bitmap(file))
                    {
                        predictionLabel.Text = controller.RecogniseImage(bmp).ToString();

                        pictureBox1.Image = bmp;
                        pictureBox1.Invalidate();
                    }
                }
                catch (IOException)
                {
                }
            }
        }

        private void CropButton_Click(object sender, EventArgs e)
        {
            ImageProcessing.SaveImage(ImageProcessing.CropWhite(pictureBox1.Image, 75, 75), Environment.CurrentDirectory + "\\image.jpg");
        }
    }
}
