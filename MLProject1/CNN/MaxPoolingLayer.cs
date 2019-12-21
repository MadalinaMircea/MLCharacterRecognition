using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class MaxPoolingLayer : NetworkLayer
    {
        public int PoolWidth { get; }
        public int PoolHeight { get; }
        public MaxPoolingLayer() : base("MaxPooling")
        {
            PoolWidth = 2;
            PoolHeight = 2;
        }

        [JsonConstructor]
        public MaxPoolingLayer(int poolWidth, int poolHeight) : base("MaxPooling")
        {
            PoolWidth = poolWidth;
            PoolHeight = poolHeight;
        }
    }
}
