using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    class Weight
    {
        public Unit FromUnit { get; }
        public int Value { get; }

        public Weight(Unit fromUnit, int value)
        {
            FromUnit = fromUnit;
            Value = value;
        }
    }
}
