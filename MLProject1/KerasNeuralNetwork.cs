using Keras;
using Keras.Callbacks;
using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.PreProcessing.Image;
using Keras.Utils;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1
{
    public class KerasNeuralNetwork
    {
        BaseModel model;

        public KerasNeuralNetwork(string modelFile, string weightsFile)
        {
            Sequential newModel = CreateModel3();
            newModel = FitAndEvaluate(newModel);
            WriteModelToFiles(newModel);
            //model = LoadModel(modelFile, weightsFile);
            model = LoadModel("modelJson5.json", "best_weights5.h5");
            //foreach (double d in EvaluateModel())
            //{
            //    Console.WriteLine(d);
            //}
            //model = LoadModel(Environment.CurrentDirectory + "\\" + modelFile, Environment.CurrentDirectory + "\\" + weightsFile);
        }
        private char GetPrediction(NDarray predictions)
        {
            int maxx = 0, maxi = 0;
            for (int i = 0; i < predictions.size; i++)
            {
                if ((bool)(predictions[i] > maxx))
                {
                    maxi = i;
                }
            }

            return (char)(maxi + 65);
        }

        private void WriteModelToFiles(Sequential newModel)
        {
            //serialize model to JSON
            string modelJson = newModel.ToJson();
            //File.WriteAllText("modelJson2.json", modelJson);
            ////serialize weights to HDF5
            //newModel.SaveWeight("modelWeights2.h5");

            File.WriteAllText("modelJson5.json", modelJson);
            //serialize weights to HDF5
            newModel.SaveWeight("modelWeights5.h5");
        }

        private Sequential CreateModel()
        {
            int imgWidth = 100;
            int imgHeight = 75;

            Shape inputShape;

            if (Backend.ImageDataFormat() == "channels_first")
                inputShape = new Shape(3, imgWidth, imgHeight);
            else
                inputShape = new Shape(imgWidth, imgHeight, 3);

            Sequential newModel = new Sequential();
            newModel.Add(new Conv2D(32, new Tuple<int, int>(5, 5), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Flatten());
            newModel.Add(new Dense(128, activation: "relu"));
            newModel.Add(new Dense(26, activation: "softmax"));

            newModel.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return newModel;
        }

        private Sequential CreateModel2()
        {
            int imgWidth = 100;
            int imgHeight = 75;

            Shape inputShape;

            if (Backend.ImageDataFormat() == "channels_first")
                inputShape = new Shape(3, imgWidth, imgHeight);
            else
                inputShape = new Shape(imgWidth, imgHeight, 3);

            Sequential newModel = new Sequential();
            newModel = new Sequential();
            newModel.Add(new Conv2D(32, new Tuple<int, int>(28, 28), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Conv2D(64, new Tuple<int, int>(14, 14), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Conv2D(128, new Tuple<int, int>(7, 7), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Flatten());
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Dense(512, activation: "relu"));
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Dense(256, activation: "relu"));
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Dense(50, activation: "relu"));
            newModel.Add(new Dense(26, activation: "softmax"));

            newModel.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return newModel;
        }

        private Sequential CreateModel3()
        {
            int imgWidth = 100;
            int imgHeight = 75;

            Shape inputShape;

            if (Backend.ImageDataFormat() == "channels_first")
                inputShape = new Shape(3, imgHeight, imgWidth);
            else
                inputShape = new Shape(imgHeight, imgWidth, 3);

            Sequential newModel = new Sequential();
            newModel.Add(new Conv2D(32, new Tuple<int, int>(10, 10), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Conv2D(64, new Tuple<int, int>(10, 10), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Conv2D(128, new Tuple<int, int>(5, 5), input_shape: inputShape, activation: "relu"));
            newModel.Add(new Flatten());
            newModel.Add(new Dense(512, activation: "relu"));
            newModel.Add(new Dropout(0.5));
            newModel.Add(new Dense(26, activation: "softmax"));

            newModel.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return newModel;
        }

        private Sequential CreateModel4()
        {
            int imgWidth = 75;
            int imgHeight = 75;

            Shape inputShape;

            if (Backend.ImageDataFormat() == "channels_first")
                inputShape = new Shape(3, imgHeight, imgWidth);
            else
                inputShape = new Shape(imgHeight, imgWidth, 3);

            Sequential newModel = new Sequential();
            newModel.Add(new Conv2D(32, new Tuple<int, int>(10, 10), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Conv2D(64, new Tuple<int, int>(10, 10), input_shape: inputShape, activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Conv2D(128, new Tuple<int, int>(5, 5), input_shape: inputShape, activation: "relu"));
            newModel.Add(new Flatten());
            newModel.Add(new Dense(512, activation: "relu"));
            newModel.Add(new Dropout(0.5));
            newModel.Add(new Dense(26, activation: "softmax"));

            newModel.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return newModel;
        }

        private Sequential FitAndEvaluate(Sequential newModel)
        {
            int imgWidth = 100;
            int imgHeight = 75;
            int trainSamples = 31980;
            int testSamples = 4420;
            int epochs = 50;
            int batchSize = 50;

            string trainDirectory = "data/new/Alphabet Training";
            string testDirectory = "data/new/Alphabet Testing";
            string validationDirectory = "data/new/Alphabet Validation";

            ImageDataGenerator generator = new ImageDataGenerator(rescale: (float)(1.00 / 255.00));

            //load and iterate training dataset
            KerasIterator trainIterator = generator.FlowFromDirectory(trainDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgWidth),
                                                         batch_size: batchSize);

            // load and iterate testing dataset
            KerasIterator testIterator = generator.FlowFromDirectory(testDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgWidth),
                                                         batch_size: batchSize);

            KerasIterator validationIterator = generator.FlowFromDirectory(validationDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgWidth),
                                                         batch_size: batchSize);

            ModelCheckpoint checkpoint = new ModelCheckpoint(filepath: "best_weights5.h5",
                                           monitor: "val_accuracy",
                                           verbose: 1,
                                           save_best_only: true);

            newModel.FitGenerator(trainIterator, steps_per_epoch: trainSamples / batchSize, epochs: epochs, verbose: 1,
                validation_data: validationIterator, callbacks: new Callback[] { checkpoint });

            //evaluate model
            double[] loss = newModel.EvaluateGenerator(testIterator, steps: testSamples / batchSize);

            foreach (double d in loss)
            {
                Console.Write(d + " ");
            }

            return newModel;
        }

        private BaseModel LoadModel(string modelPath, string weightsPath)
        {
            string text = File.ReadAllText(modelPath);
            BaseModel model = Model.ModelFromJson(text);
            model.LoadWeight(weightsPath);

            model.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return model;
        }

        public double[] EvaluateModel()
        {
            int testSamples = 4160;
            int batchSize = 30;
            int imgWidth = 100;
            int imgHeight = 75;

            string testDirectory = "data/Alphabet Testing";

            ImageDataGenerator generator = new ImageDataGenerator(rescale: (float)(1.00 / 255.00));

            // load and iterate testing dataset
            KerasIterator testIterator = generator.FlowFromDirectory(testDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgWidth),
                                                         batch_size: batchSize);

            return model.EvaluateGenerator(testIterator, steps: testSamples / batchSize);
        }

        public char RecogniseImage(string path)
        {
            var img = ImageUtil.LoadImg(path, target_size: new Shape(75, 100));
            NDarray arr = ImageUtil.ImageToArray(img);
            arr = Numpy.np.expand_dims(arr, 0);

            NDarray response = model.Predict(arr);

            return GetPrediction(response[0]);
        }
    }
}
