using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public class ImageRepository
    {
        public List<InputOutputPair> TrainingSetPaths { get; set; }
        public List<InputOutputPair> TestingSetPaths { get; set; }
        public List<InputOutputPair> ValidationSetPaths { get; set; }
    }
}
