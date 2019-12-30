using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLProject1.CNN
{
    public class NetworkLayerSpecificConverter : DefaultContractResolver
    {
        protected override JsonConverter ResolveContractConverter(Type objectType)
        {
            if (typeof(NetworkLayer).IsAssignableFrom(objectType) && !objectType.IsAbstract)
                return null;
            return base.ResolveContractConverter(objectType);
        }
    }

    public class NetworkLayerConverter : JsonConverter
    {
        static JsonSerializerSettings SpecifiedSubclassConversion = new JsonSerializerSettings() { ContractResolver = new NetworkLayerSpecificConverter() };

        public override bool CanConvert(Type objectType)
        {
            return (objectType == typeof(NetworkLayer));
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            JObject jo = JObject.Load(reader);
            switch (jo["Type"].Value<string>())
            {
                case "Convolutional":
                    return JsonConvert.DeserializeObject<ConvolutionalLayer>(jo.ToString(), SpecifiedSubclassConversion);
                case "MaxPooling":
                    return JsonConvert.DeserializeObject<MaxPoolingLayer>(jo.ToString(), SpecifiedSubclassConversion);
                case "Dropout":
                    return JsonConvert.DeserializeObject<DropoutLayer>(jo.ToString(), SpecifiedSubclassConversion);
                case "Flatten":
                    return JsonConvert.DeserializeObject<FlattenLayer>(jo.ToString(), SpecifiedSubclassConversion);
                case "Dense":
                    return JsonConvert.DeserializeObject<DenseLayer>(jo.ToString(), SpecifiedSubclassConversion);
                case "Input":
                    return JsonConvert.DeserializeObject<InputLayer>(jo.ToString(), SpecifiedSubclassConversion);
                default:
                    throw new Exception();
            }
            throw new NotImplementedException();
        }

        public override bool CanWrite
        {
            get { return false; }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            throw new NotImplementedException();
        }
    }

    public class ToStringJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return true;
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            writer.WriteValue(value.ToString());
        }

        public override bool CanRead
        {
            get { return false; }
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw new NotImplementedException();
        }
    }
}
