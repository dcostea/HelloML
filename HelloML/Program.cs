using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using static System.Console;

namespace HelloML
{
    class Program
    {
        private static MLContext Context { get; set; }
        private static ITransformer Model { get; set; }

        static void Main(string[] args)
        {
            const string sampleCsv = "sample.csv";

            Context = new MLContext(seed: 0);

            // get training data
            var reader = Context.Data.CreateTextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("MinHour", DataKind.R4, 0),
                    new TextLoader.Column("MaxHour", DataKind.R4, 1),
                    new TextLoader.Column("Label", DataKind.TX, 2)
                },
            });
            WriteLine("Loading training data...");
            var trainingDataView = reader.Read(sampleCsv);

            // train the model
            var trainingPipeline = Context.Transforms.Concatenate("Features", "MinHour", "MaxHour")
                .Append(Context.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.Everything)
                .AppendCacheCheckpoint(Context)
                //.Append(Context.MulticlassClassification.Trainers.LogisticRegression())
                .Append(Context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
                .Append(Context.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "PredictedLabel")));

            WriteLine("Training the model...");
            Model = trainingPipeline.Fit(trainingDataView);

            // predict data
            WriteLine("Predict some data...");
            var sample1 = new SourceData
            {
                MinHour = 22,
                MaxHour = 23
            };
            WriteLine($"Predict greeting for interval {sample1.MinHour} - {sample1.MaxHour}: {Predict(sample1).PredictedLabel}");

            var sample2 = new SourceData
            {
                MinHour = 9,
                MaxHour = 11
            };
            WriteLine($"Predict greeting for interval {sample2.MinHour} - {sample2.MaxHour}: {Predict(sample2).PredictedLabel}");

            var sample3 = new SourceData
            {
                MinHour = 15,
                MaxHour = 16
            };
            WriteLine($"Predict greeting for interval {sample3.MinHour} - {sample3.MaxHour}: {Predict(sample3).PredictedLabel}");


            WriteLine("\nPress any key to continue...");
            ReadKey(true);
        }

        static SourcePrediction Predict(SourceData data)
        {
            var predictionEngine = Model.CreatePredictionEngine<SourceData, SourcePrediction>(Context);
            return predictionEngine.Predict(data);
        }
    }

    public class SourceData
    {
        public float MinHour { get; set; }

        public float MaxHour { get; set; }

        [ColumnName("Label")]
        public string Label;
    }

    class SourcePrediction
    {
        public string PredictedLabel { get; set; }
    }
}
