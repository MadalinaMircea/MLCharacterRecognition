using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class ImageController
    {
        public ImageRepository Repo { get; set; }

        public ImageController(ImageRepository repo)
        {
            Repo = repo;
        }

        private List<InputOutputPair> ReadFromDirectory(string directory)
        {
            List<InputOutputPair> result = new List<InputOutputPair>();

            string[] directories = Directory.GetDirectories(directory);

            Task[] tasks = new Task[directories.Length];

            for(int i = 0; i < directories.Length; i++)
            {
                int newI = 0 + i;
                tasks[i] = Task.Run(() =>
                {
                    foreach (string input in Directory.GetFiles(directories[newI]))
                    {
                        InputOutputPair pair = new InputOutputPair(input, directories[newI]);
                        Monitor.Enter(result);
                        result.Add(pair);
                        Monitor.Exit(result);
                    }
                });
                
            }

            Task.WaitAll(tasks);

            return result;
        }
        public void ReadSet(string set, string directory)
        {
            switch(set)
            {
                case "train":
                    Repo.TrainingSetPaths = ReadFromDirectory(directory);
                    break;
                case "test":
                    Repo.TestingSetPaths = ReadFromDirectory(directory);
                    break;
                case "valid":
                    Repo.ValidationSetPaths = ReadFromDirectory(directory);
                    break;

            }
        }

        private List<InputOutputPair> Shuffle(List<InputOutputPair> set)
        {
            for(int i = 0; i < set.Count; i++)
            {
                int x = GlobalRandom.GetRandomInt(i, set.Count);
                InputOutputPair aux = set[x];
                set[x] = set[i];
                set[i] = aux;
            }

            return set;
        }
        public void ShuffleSets()
        {
            Task t1 = Task.Run(() => { Shuffle(Repo.TrainingSetPaths); });
            Task t2 = Task.Run(() => { Shuffle(Repo.TestingSetPaths); });
            Task t3 = Task.Run(() => { Shuffle(Repo.ValidationSetPaths); });

            t1.Wait();
            t2.Wait();
            t3.Wait();
        }
    }
}
