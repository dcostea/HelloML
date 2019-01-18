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

            Context = new MLContext(seed: 1);

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
            WriteLine("Loading data...");
            var data = reader.Read(sampleCsv);
            var (trainingDataView, testingDataView) = Context.MulticlassClassification.TrainTestSplit(data, testFraction: 0.2, seed: 1);

            // train the model
            var trainingPipeline = Context.Transforms.Concatenate("Features", "MinHour", "MaxHour")
                .Append(Context.Transforms.Conversion.MapValueToKey("Label"), TransformerScope.Everything)
                .Append(Context.MulticlassClassification.Trainers.LogisticRegression())
                //.AppendCacheCheckpoint(Context)
                //.Append(Context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
                .Append(Context.Transforms.Conversion.MapKeyToValue(("PredictedLabel", "PredictedLabel")));

            WriteLine("Training the model...");
            Model = trainingPipeline.Fit(trainingDataView);

            var metrics = Context.MulticlassClassification.Evaluate(Model.Transform(testingDataView), "Label", "Score");
            PrintMultiClassClassificationMetrics("", metrics);

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

        public static void PrintMultiClassClassificationMetrics(string name, MultiClassClassifierMetrics metrics)
        {
            WriteLine($"************************************************************");
            WriteLine($"*    Metrics for {name} multi-class classification model   ");
            WriteLine($"*-----------------------------------------------------------");
            WriteLine($"    AccuracyMacro = {metrics.AccuracyMacro:0.####}, a value between 0 and 1, the closer to 1, the better");
            WriteLine($"    AccuracyMicro = {metrics.AccuracyMicro:0.####}, a value between 0 and 1, the closer to 1, the better");
            WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            WriteLine($"************************************************************");
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
