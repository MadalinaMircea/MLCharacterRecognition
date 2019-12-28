using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class FilteredImage : LayerOutput
    {
        public int NumberOfChannels { get; set; }
        public FilteredImageChannel[] Channels { get; set; }

        public FilteredImage(int numberOfChannels, FilteredImageChannel[] channels)
        {
            NumberOfChannels = numberOfChannels;
            Channels = channels;
            NumberOfWeights = numberOfChannels;
        }
    }
}
