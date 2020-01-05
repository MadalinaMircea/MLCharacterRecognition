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

        public int Size { get; set; }

        public FilteredImage(int numberOfChannels, FilteredImageChannel[] channels)
        {
            NumberOfChannels = numberOfChannels;
            Channels = channels;

            if (channels.Length == 0)
            {
                Size = 0;
            }
            else
            {
                Size = channels[0].Size;
            }
        }

        public FilteredImage(int numberOfChannels, int channelSize)
        {
            NumberOfChannels = numberOfChannels;
            Channels = new FilteredImageChannel[numberOfChannels];
            Size = channelSize;
        }

        public override double Sum()
        {
            double result = 0;

            for(int c = 0; c < NumberOfChannels; c++)
            {
                result += MatrixUtils.ElementSum(Channels[c].Values);
            }

            return result;
        }
    }
}
