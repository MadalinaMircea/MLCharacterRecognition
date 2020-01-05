using Keras;
using Keras.Callbacks;
using Keras.Layers;
using Keras.Models;
using Keras.PreProcessing.Image;
using Numpy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1
{
    class KerasController
    {
        BaseModel model;
        public KerasController(string modelFile, string weightsFile)
        {
            //Sequential newModel = CreateModel();
            //model = newModel;
            //WriteModelToFiles(modelFile, weightsFile);
            //model = FitAndEvaluate(newModel, weightsFile);
            model = LoadModel(modelFile, weightsFile);
        }

        private void WriteModelToFiles(string modelFile, string weightFile)
        {
            //serialize model to JSON
            string modelJson = model.ToJson();
            //File.WriteAllText("modelJson2.json", modelJson);
            ////serialize weights to HDF5
            //newModel.SaveWeight("modelWeights2.h5");

            File.WriteAllText(modelFile, modelJson);
            //serialize weights to HDF5
            model.SaveWeight(weightFile);
        }

        private Sequential CreateModel()
        {
            int imgHeight = 75;

            Shape inputShape;

            if (Backend.ImageDataFormat() == "channels_first")
                inputShape = new Shape(3, imgHeight, imgHeight);
            else
                inputShape = new Shape(imgHeight, imgHeight, 3);

            Sequential newModel = new Sequential();
            newModel.Add(new Conv2D(5, new Tuple<int, int>(5, 5), input_shape: inputShape, padding: "same", activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Conv2D(5, new Tuple<int, int>(3, 3), input_shape: inputShape, padding: "same", activation: "relu"));
            newModel.Add(new MaxPooling2D());
            newModel.Add(new Dropout(0.2));
            newModel.Add(new Flatten());
            newModel.Add(new Dropout(0.5));
            newModel.Add(new Dense(26, activation: "softmax"));

            newModel.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return newModel;
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

        private BaseModel LoadModel(string modelPath, string weightsPath)
        {
            string text = File.ReadAllText(modelPath);
            BaseModel model = Model.ModelFromJson(text);
            model.LoadWeight(weightsPath);

            model.Compile(loss: "categorical_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });

            return model;
        }

        public char RecogniseImage(string path)
        {
            var img = ImageUtil.LoadImg(path, target_size: new Shape(75, 75));
            NDarray arr = ImageUtil.ImageToArray(img);

            arr = Numpy.np.expand_dims(arr, 0);

            NDarray response = model.Predict(arr);

            char pred = GetPrediction(response[0]);

            return pred;
        }

        private Sequential FitAndEvaluate(Sequential newModel, string bestWeightsFile)
        {
            int imgHeight = 75;
            int trainSamples = 31980;
            int testSamples = 4420;
            int epochs = 20;
            int batchSize = 26;

            string trainDirectory = "E:\\Projects\\MLProject1\\MLProject1\\bin\\Debug\\data\\Train";
            string testDirectory = "E:\\Projects\\MLProject1\\MLProject1\\bin\\Debug\\data\\Test";
            string validationDirectory = "E:\\Projects\\MLProject1\\MLProject1\\bin\\Debug\\data\\Valid";

            ImageDataGenerator generator = new ImageDataGenerator(rescale: (float)(1.00 / 255.00));

            //load and iterate training dataset
            KerasIterator trainIterator = generator.FlowFromDirectory(trainDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgHeight),
                                                         batch_size: batchSize);
            //color_mode:"grayscale");

            // load and iterate testing dataset
            KerasIterator testIterator = generator.FlowFromDirectory(testDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgHeight),
                                                         batch_size: batchSize);
            //color_mode: "grayscale");

            KerasIterator validationIterator = generator.FlowFromDirectory(validationDirectory,
                                                         class_mode: "categorical",
                                                         target_size: new Tuple<int, int>(imgHeight, imgHeight),
                                                         batch_size: batchSize);

            ModelCheckpoint checkpoint = new ModelCheckpoint(filepath: bestWeightsFile,
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
    }
}
