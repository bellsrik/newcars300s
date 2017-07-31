package com.cars.bigdata.optimaltimetobuy

/**
  * Created by sbellary on 5/21/2017.
  */

//IMPORT SPARK LIBRARIES
import org.apache.avro.Foo
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext, types}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
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
import org.apache.spark.sql.functions.udf
import java.util.concurrent.TimeUnit
import java.util.Date
import java.text.SimpleDateFormat

//SINGLETON OBJECT GBT REGRESSION FOR OPTIMAL TIME TO BUY PROJECT
object GBTRegression {

  org.apache.spark.sql.catalyst.encoders.OuterScopes.addOuterScope(this)
  case class ActiveVehicleList(val VEHICLE_ID: String, val PRED_DAYS: String, val PREDICTED_DATE: String)
  case class KeyValuePair(vehicle_id:String, prediction:List[ActiveVehicleList])
  //DEFINE MAIN METHOD FOR GBT REGRESSION
  def main(args: Array[String]): Unit = {
    //SETUP LOGGING
    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
    //SETUP CONFIG FACTOR FOR PRODUCTION DEPLOYMENT
    val appConf = ConfigFactory.load()
    val conf = new SparkConf()
      .setAppName("GradientBoostRegression")
      .setMaster(appConf.getConfig(args(0)).getString("deploymentMaster"))
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.registerKryoClasses(Array(classOf[Foo]))

    //SETUP SPARK CONTEXT AND SQL CONTEXT
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //VALIDATE INPUT AND OUTPUT PATHS EXISTS
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val trainDataInputPathExists = fs.exists(new Path(appConf.getConfig(args(1)).getString("trainDataInputPath")))
    val outputPathExists = fs.exists(new Path(appConf.getConfig(args(2)).getString("outputPath")))
    val gbtModelPathExists = fs.exists(new Path(appConf.getConfig(args(3)).getString("gbtModelPath")))

    //VALIDATE TRAIN DATA INPUT PATH
    if(!trainDataInputPathExists) {
      println("Invalid training data input path")
      return
    }

    //VALIDATE AND OVERWRITE OUTPUT
    if(outputPathExists)
      fs.delete(new Path(appConf.getConfig(args(2)).getString("outputPath")), true)

    //VALIDATE AND OVERWRITE GBT MODEL OUTPUT
    if(gbtModelPathExists)
      fs.delete(new Path(appConf.getConfig(args(3)).getString("gbtModelPath")), true)

    //RECORD THE START-TIME FOR THE PROGRAM
    val start_time_main = Calendar.getInstance().getTime()

    //LOAD DATA, REMOVE HEADER, RUN IT THROUGH MAPPER FUNCTION AND CREATE DATA FRAME
    val csvData = sc.textFile(appConf.getConfig(args(1)).getString("trainDataInputPath"))
    val header = csvData.first()
    val filteredRDD = csvData.filter(row => row != header)
    val rowRDD = filteredRDD.map(trainDataMapper)
    val trainDFrame = sqlContext.createDataFrame(rowRDD)
    trainDFrame.registerTempTable("tDF")

    //ADD A NEW COLUMN/FEATURE TO OUR DATA FRAME "PRICE_RATIO" AND CREATE MUTABLE DATA FRAME "DF"

    var df = sqlContext.sql(
      """SELECT vehicle_id,
        |dma_code,
        |mileage,
        |price,
        |photo_count,
        |days,
        |vdp_views,
        |total_vehicles,
        |make_id,
        |make_model_id,
        |model_year,
        |trim_id,
        |current_date() AS current_date
        |FROM tDF
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

    //RECORD THE START-TIME FOR APPENDING AND VECTOR ASSEMBLY
    val start_time_train = Calendar.getInstance().getTime()

    //PERFORM DATA TYPE TRANSFORMATIONS FROM STRING TO DOUBLE
    df = df.withColumn("mileage_int", df("mileage").cast("double"))
    df = df.withColumn("price_int", df("price").cast("double"))
    df = df.withColumn("photo_count_int", df("photo_count").cast("double"))
    df = df.withColumn("vdp_views_int", df("vdp_views").cast("double"))
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
      "vdp_views_int",
      "days_int"
    )
    df = df.na.fill(0)
    df.show()

    //ASSEMBLE ALL OUR FEATURES AND CREATE "FEATURES" VECTOR USING "VECTOR ASSEMBLER"
    val assembler = new VectorAssembler()
          .setInputCols(Array("dma_sparse_vect",
            "make_sparse_vect",
            "make_model_sparse_vect",
            "model_year_sparse_vect",
            "trim_id_sparse_vect",
            "mileage_int",
            "price_int",
            "photo_count_int",
            "vdp_views_int"
            )
          )
          .setOutputCol("features")
    val assemblerData = assembler.transform(df)
    assemblerData.show()
    val assemblerArray = assemblerData.randomSplit(Array(0.7, 0.3))
    val trainingData = assemblerArray(0).cache()
    val testData = assemblerArray(1)

    //MAKE THE GBT REGRESSION MODEL USING "GBTREGRESSOR"
    val gbt = new GBTRegressor()
      .setFeaturesCol("features")
      .setLabelCol("days_int")
      .setMaxIter(appConf.getConfig(args(4)).getInt("iterations"))
      .setMaxDepth(appConf.getConfig(args(5)).getInt("depth"))
      .setStepSize(0.001)
      .setMinInstancesPerNode(3)

    //TRAIN THE GBT MODEL WITH TRAINING DATA
    val gbtModel = gbt.fit(trainingData)

    sc.parallelize(Seq(gbtModel)).saveAsObjectFile(appConf.getConfig(args(3)).getString("gbtModelPath"))

    //GET THE TIME TO TRAIN THE MODEL
    val end_time_train = Calendar.getInstance().getTime()
    val elapsed_time_train =  ((end_time_train.getTime() - start_time_train.getTime())/1000).toString
    println("Time taken to train the model: " + elapsed_time_train + " seconds.")

    //PREDICTIONS FOR TEST DATA
    var predictionsTest = gbtModel.transform(testData)

    //SHOW OR SAVE RESULTS FOR TEST DATA
    predictionsTest = predictionsTest.select("vehicle_id", "days_int", "current_date", "prediction")
    predictionsTest = predictionsTest.withColumn("pred_days", predictionsTest("prediction").cast("Int"))
    predictionsTest.show

/*    import sqlContext.implicits._
    import org.apache.spark.sql._
    import breeze.math.Complex
    import breeze.numerics._
    import breeze.stats.median
    import breeze.numerics.abs

    val predicted = predictionsTest.col("pred_days") //List[Double]( ... )
    val days_infer_sale = predictionsTest.col("days_int") //List[Double]( ... )
    val result_col = predicted - days_infer_sale
    println("median absolute = " + median(abs(predicted)))*/

    val date_add = udf((x: String, y: Int) => {
      val sdf = new SimpleDateFormat("yyyy-MM-dd")
      val result = new Date(sdf.parse(x).getTime() + TimeUnit.DAYS.toMillis(y))
      sdf.format(result)
    } )
    import sqlContext.implicits._
    val dx = predictionsTest.select("vehicle_id", "current_date", "pred_days")
    val dy = dx.withColumn("predicted_date", date_add($"current_date", $"pred_days"))
    dy.show()

    dy.registerTempTable("disttable")
    val query = sqlContext.sql("select * from disttable")

    val rows: RDD[Row] = query.rdd //: RDD[Row]
    val sim_veh = rows.groupBy(rw => rw.getString(0)).map(row => {
      var vlist = List[ActiveVehicleList]()
      val vehicle_id = row._1
      val sim_vehicle_id = row._2.foreach(row =>
        vlist = ActiveVehicleList(row(0).toString, row(2).toString,row(3).toString) :: vlist
      )
      KeyValuePair(vehicle_id,vlist)
    }
    )

    sqlContext.createDataFrame(sim_veh).registerTempTable("finaltable")

    val finaldf = sqlContext.sql("select * from finaltable")

    finaldf.write.format("org.apache.spark.sql.execution.datasources.json").save(appConf.getConfig(args(2)).getString("outputPath"))

    //PRINT TREE IF ITS REQUIRED
    val model = gbtModel.asInstanceOf[GBTRegressionModel]
    println("Learned regression GBT model:\n" + model.toDebugString)

    //EVALUATE RMSE FOR "TEST" DATA USING REGRESSION EVALUATOR
    val testRMSE = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    //PRINT EVALUATION RESULTS FOR TEST DATA
    val rmse_test = testRMSE.evaluate(predictionsTest)
    println("Root Mean Squared Error (RMSE) on TEST data = " + rmse_test)

    //EVALUATE MAE FOR "TEST" DATA USING REGRESSION EVALUATOR
    val testMAE = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    //PRINT EVALUATION RESULTS FOR TEST DATA
    val mae_test = testMAE.evaluate(predictionsTest)
    println("Mean Absolute Error (MAE) on TEST data = " + mae_test)

    //EVALUATE R-SQUARED FOR "TEST" DATA USING REGRESSION EVALUATOR
    val testRSquared = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    //PRINT EVALUATION RESULTS FOR TEST DATA
    val rsquared_test = testRSquared.evaluate(predictionsTest)
    println("R-Squared on TEST data = " + rsquared_test)

    //EVALUATE MSE FOR "TEST" DATA USING REGRESSION EVALUATOR
    val testMSE = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("mse")

    //PRINT EVALUATION RESULTS FOR TEST DATA
    val mse_test = testMSE.evaluate(predictionsTest)
    println("Mean Squared Error (MSE) on TEST data = " + mse_test)

    //GET THE TOTAL TIME TO RUN THE PROGRAM
    val end_time_main = Calendar.getInstance().getTime()
    val elapsed_time_main =  ((end_time_main.getTime() - start_time_main.getTime())/1000).toString
    println("Total time to run the program : " + elapsed_time_main + " seconds.")

    //STOP SPARK CONTEXT
    sc.stop()
  }
}
