using HDF5DotNet;
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
    public partial class Form1 : Form
    {
        Point lastPoint = Point.Empty;

        //used to see if the user is clicking or not
        bool isMouseDown = false;

        //KerasNeuralNetwork network;
        CNNController controller = new CNNController();

        int i = 0;

        public Form1()
        {
            InitializeComponent();
            //network = new KerasNeuralNetwork("modelJson2.json", "modelWeights2.h5");
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

            //controller.CreateAndCompileModel("tryNewJson.json", "tryWeights");

            controller.PrepareImageSets("new\\Square\\Alphabet Training",
                "new\\Square\\Alphabet Testing", "new\\Square\\Alphabet Validation");

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
            string path = Environment.CurrentDirectory + "\\image.jpg";

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
            DialogResult result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
            {
                string file = openFileDialog1.FileName;
                try
                {
                    predictionLabel.Text = controller.RecogniseImage(new Bitmap(file)).ToString();

                    Bitmap bmp = new Bitmap(file);
                    pictureBox1.Image = bmp;
                    pictureBox1.Invalidate();
                }
                catch (IOException)
                {
                }
            }
        }

        private void CropButton_Click(object sender, EventArgs e)
        {
            //ImageProcessing.CropWhite(pictureBox1.Image, Environment.CurrentDirectory +, 100, 75);
        }
    }
}
