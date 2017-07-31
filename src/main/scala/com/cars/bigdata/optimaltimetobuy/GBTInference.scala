package com.cars.bigdata.optimaltimetobuy

/**
  * Created by sbellary on 6/5/2017.
  */

//IMPORT SPARK LIBRARIES
import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
//IMPORT SPARK ML LIBRARIES
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.regression.GBTRegressionModel
//IMPORT OTHER UTILITY LIBRARIES
import org.apache.log4j._
import org.apache.hadoop.fs._
import java.util.Calendar
import com.typesafe.config.ConfigFactory
import com.cars.bigdata.optimaltimetobuy.RowMapper._
import com.cars.bigdata.optimaltimetobuy.GBTRegression._
import org.apache.spark.sql.functions.udf
import java.util.concurrent.TimeUnit
import java.util.Date
import java.text.SimpleDateFormat

object GBTInference {

  //DEFINE MAIN METHOD FOR GBT REGRESSION
  def main(args: Array[String]): Unit = {
    //SETUP LOGGING
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")

    //SETUP CONFIG FACTOR FOR PRODUCTION DEPLOYMENT
    val appConf = ConfigFactory.load()
    val conf = new SparkConf()
      .setAppName("GBTR Inference")
      .setMaster(appConf.getConfig(args(0)).getString("deploymentMaster"))

    //SETUP SPARK CONTEXT AND SQL CONTEXT
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //VALIDATE INPUT AND OUTPUT PATHS EXISTS
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val vehicleDailyInputPathExists = fs.exists(new Path(appConf.getConfig(args(1)).getString("vehicleDailyInputPath")))
    val outputPathExists = fs.exists(new Path(appConf.getConfig(args(2)).getString("outputPath")))
    val gbtModelPathExists = fs.exists(new Path(appConf.getConfig(args(3)).getString("gbtModelPath")))

    //VALIDATE VEHICLE DAILY INPUT PATH
    if(!vehicleDailyInputPathExists) {
      println("Invalid vehicle daily input path")
      return
    }
    //VALIDATE AND OVERWRITE OUTPUT
    if(outputPathExists)
      fs.delete(new Path(appConf.getConfig(args(2)).getString("outputPath")), true)

    //VALIDATE GBT MODEL INPUT PATH
    if(!gbtModelPathExists) {
      println("Invalid GBT Model input path")
      return
    }

    //RECORD THE START-TIME FOR THE PROGRAM
    val start_time_main = Calendar.getInstance().getTime()

    //LOAD DATA, REMOVE HEADER, RUN IT THROUGH MAPPER FUNCTION AND CREATE DATA FRAME
    val vehicleDailyData = sc.textFile(appConf.getConfig(args(1)).getString("vehicleDailyInputPath"))
    val header = vehicleDailyData.first()
    val filteredVehicleRDD = vehicleDailyData.filter(row => row != header)
    val rowVehicleRDD = filteredVehicleRDD.map(vehicleDailyMapper)
    val vehicleDFrame = sqlContext.createDataFrame(rowVehicleRDD)
    vehicleDFrame.registerTempTable("vDF")

    //ADD A NEW COLUMN/FEATURE TO OUR DATA FRAME "PRICE_RATIO" AND CREATE MUTABLE DATA FRAME "DF"

    var df = sqlContext.sql(
      """SELECT vehicle_id,
        |dma_code,
        |mileage,
        |price,
        |photo_count,
        |days,
        |total_vehicles,
        |make_id,
        |make_model_id,
        |model_year,
        |trim_id,
        |current_date() AS current_date
        |FROM vDF
        |WHERE days >= 1 AND days <= 180
        |AND make_id IN ("20032O",
        |"20025O",
        |"20061O",
        |"20012O",
        |"20001O",
        |"20041O",
        |"20019O",
        |"20005O",
        |"20024O",
        |"20035O",
        |"20081O",
        |"20088O",
        |"20049O",
        |"20016O",
        |"20077O",
        |"20064O",
        |"20070O",
        |"20017O",
        |"20089O",
        |"20038O",
        |"20085O",
        |"20066O",
        |"20073O",
        |"20042O",
        |"20030O",
        |"20074O",
        |"20068O",
        |"20075O",
        |"20053O",
        |"44763O",
        |"20052O",
        |"20028O",
        |"20015O",
        |"20044O",
        |"20008O",
        |"20006O",
        |"20021O")""".stripMargin)

    //DROP NULLS FROM THE DATA FRAME/FILL ZEROES FOR NULLS
    df = df.na.fill(0)
    df.show(10)

    //RECORD THE START-TIME FOR INDEXING AND ENCODING
    val start_time_idx = Calendar.getInstance().getTime()

    //CREATE INDEXES AND ON-HOT ENCODED VECTORS FOR CATEGORICAL FEATURES
    //INDEX AND ENCODE DMA_CODE
    val dma_code_indexer = new StringIndexer().setInputCol("dma_code").setOutputCol("dma_code_IDX")
    df = dma_code_indexer.fit(df).transform(df)
    val dma_code_encoder = new OneHotEncoder().setInputCol("dma_code_IDX").setOutputCol("dma_sparse_vect")
    df = dma_code_encoder.transform(df)

    //INDEX AND ENCODE MAKE_ID
    val make_id_indexer = new StringIndexer().setInputCol("make_id").setOutputCol("makeIDX")
    df = make_id_indexer.fit(df).transform(df)
    val make_id_encoder = new OneHotEncoder().setInputCol("makeIDX").setOutputCol("make_sparse_vect")
    df = make_id_encoder.transform(df)

    //INDEX AND ENCODE MAKE_MODEL_ID
    val make_model_indexer = new StringIndexer().setInputCol("make_model_id").setOutputCol("make_model_id_IDX")
    df = make_model_indexer.fit(df).transform(df)
    val make_model_encoder = new OneHotEncoder().setInputCol("make_model_id_IDX").setOutputCol("make_model_sparse_vect")
    df = make_model_encoder.transform(df)

    //INDEX AND ENCODE MODEL_YEAR
    val model_year_indexer = new StringIndexer().setInputCol("model_year").setOutputCol("model_year_IDX")
    df = model_year_indexer.fit(df).transform(df)
    val model_year_encoder = new OneHotEncoder().setInputCol("model_year_IDX").setOutputCol("model_year_sparse_vect")
    df = model_year_encoder.transform(df)

    //INDEX AND ENCODE TRIM_ID
    val trim_id_indexer = new StringIndexer().setInputCol("trim_id").setOutputCol("trim_id_IDX")
    df = trim_id_indexer.fit(df).transform(df)
    val trim_id_encoder = new OneHotEncoder().setInputCol("trim_id_IDX").setOutputCol("trim_id_sparse_vect")
    df = trim_id_encoder.transform(df)

    //GET THE TIME TO RUN INDEXING AND ENCODING
    val end_time_idx = Calendar.getInstance().getTime()
    val elapsed_time_idx =  ((end_time_idx.getTime() - start_time_idx.getTime())/1000).toString
    println("Time taken for Indexing and Encoding: " + elapsed_time_idx + " seconds.")

    //RECORD THE START-TIME FOR TRANSFORMATIONS
    val start_time_trans = Calendar.getInstance().getTime()

    //PERFORM DATA TYPE TRANSFORMATIONS FROM STRING TO DOUBLE
    df = df.withColumn("mileage_int", df("mileage").cast("double"))
    df = df.withColumn("price_int", df("price").cast("double"))
    df = df.withColumn("photo_count_int", df("photo_count").cast("double"))
    df = df.withColumn("days_int", df("days").cast("double")).cache()
    df = df.select("vehicle_id",
      "current_date",
      "dma_sparse_vect",
      "make_sparse_vect",
      "make_model_sparse_vect",
      "model_year_sparse_vect",
      "trim_id_sparse_vect",
      "mileage_int",
      "price_int",
      "photo_count_int",
      "days_int"
    )
    df = df.na.fill(0)
    df.show()

    //GET THE TIME TO TRANSFORM THE DATA
    val end_time_trans = Calendar.getInstance().getTime()
    val elapsed_time_trans =  ((end_time_trans.getTime() - start_time_trans.getTime())/1000).toString
    println("Time taken for data prep: " + elapsed_time_trans + " seconds.")


    //RECORD THE START-TIME FOR VECTOR ASSEMBLY
    val start_time_assemble = Calendar.getInstance().getTime()

    //ASSEMBLE ALL OUR FEATURES AND CREATE "FEATURES" VECTOR USING "VECTOR ASSEMBLER"
    val vAssembler = new VectorAssembler()
      .setInputCols(Array("dma_sparse_vect",
        "make_sparse_vect",
        "make_model_sparse_vect",
        "model_year_sparse_vect",
        "trim_id_sparse_vect",
        "mileage_int",
        "price_int",
        "photo_count_int"
      )
      )
      .setOutputCol("features")

    val vAssemblerData = vAssembler.transform(df)

    vAssemblerData.show()

    //SAVE?PERSIST THE GBT MODEL

    val gbtModel = sc.objectFile[GBTRegressionModel](appConf.getConfig(args(3)).getString("gbtModelPath")).first()




/*
    // and load it back in during production
    val newGbtModel = Pipeline.load("/tmp/spark-logistic-regression-model")

    // Prepare test documents, which are unlabeled (id, text) tuples.
    val test = sqlContext.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    // Make predictions on test documents.
    model.transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }*/
    //sc.parallelize(Seq(gbtModel), 1).saveAsObjectFile("/Users/sbellary/gbt_Model_recurr")

    //val gbtModel = GBTRegressionModel.load(appConf.getConfig(args(3)).getString("gbtModelPath"))
    //val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
    /*    import org.apache.spark.ml.{Pipeline, PipelineStage}
    val pipeline = new Pipeline().setStages(stages.toArray)
    val pipelineModel =  pipeline.fit(assemblerData)
    val rfModel = pipelineModel.stages.last.asInstanceOf[GBTRegressionModel]*/
    //val gbtModel = rfModel.load(appConf.getConfig(args(3)).getString("gbtModelPath"))
    //val gbtModel = org.apache.spark.ml.regression.GBTRegressionModel.load(appConf.getConfig(args(3)).getString("gbtModelPath"))

    //GET THE TIME TO PREP THE DATA
    val end_time_assemble = Calendar.getInstance().getTime()
    val elapsed_time_assemble =  ((end_time_assemble.getTime() - start_time_assemble.getTime())/1000).toString
    println("Time taken for vector assembly: " + elapsed_time_assemble + " seconds.")

    //PREDICTIONS FOR TEST DATA
    var predictActiveListings = gbtModel.transform(vAssemblerData)

    //SHOW OR SAVE RESULTS FOR TEST DATA
    predictActiveListings = predictActiveListings.select("vehicle_id", "current_date", "prediction")
    predictActiveListings = predictActiveListings.withColumn("pred_days", predictActiveListings("prediction").cast("Int"))
    predictActiveListings.show

    val date_add = udf((x: String, y: Int) => {
      val sdf = new SimpleDateFormat("yyyy-MM-dd")
      val result = new Date(sdf.parse(x).getTime() + TimeUnit.DAYS.toMillis(y))
      sdf.format(result)
    } )

    import sqlContext.implicits._
    val dx = predictActiveListings.select("vehicle_id", "current_date", "pred_days")
    val dy = dx.withColumn("predicted_date", date_add($"current_date", $"pred_days"))
    dy.show()

    dy.coalesce(1).write.format("JSON").save(appConf.getConfig(args(2)).getString("outputPath"))

    //PRINT TREE IF ITS REQUIRED
    val model = gbtModel.asInstanceOf[GBTRegressionModel]
    println("Learned regression GBT model:\n" + model.toDebugString)

    //GET THE TOTAL TIME TO RUN THE PROGRAM
    val end_time_main = Calendar.getInstance().getTime()
    val elapsed_time_main =  ((end_time_main.getTime() - start_time_main.getTime())/1000).toString
    println("Total time to run the program : " + elapsed_time_main + " seconds.")

    //STOP SPARK CONTEXT
    sc.stop()
  }
}
