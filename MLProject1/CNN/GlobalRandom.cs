using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    static class GlobalRandom
    {
        private static Random rnd;

        public static void InitializeRandom()
        {
            rnd = new Random();
        }

        public static double GetRandomWeight()
        {
            Monitor.Enter(rnd);
            double r = rnd.NextDouble() * rnd.Next(-1, 2);
            Monitor.Exit(rnd);
            return r;
        }

        public static double GetRandomDouble()
        {
            Monitor.Enter(rnd);
            double r = rnd.NextDouble();
            Monitor.Exit(rnd);
            return r;
        }

        public static int GetRandomInt(int min, int max)
        {
            Monitor.Enter(rnd);
            int r = rnd.Next(min, max);
            Monitor.Exit(rnd);
            return r;
        }
    }
}
