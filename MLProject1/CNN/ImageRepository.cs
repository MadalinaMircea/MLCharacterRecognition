using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class ImageRepository
    {
        public List<InputOutputPair> TrainingSet { get; set; }
        public List<InputOutputPair> TestingSet { get; set; }
        public List<InputOutputPair> ValidationSet { get; set; }
    }
}
