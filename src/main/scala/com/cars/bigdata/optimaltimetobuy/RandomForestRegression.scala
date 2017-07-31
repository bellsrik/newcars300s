package com.cars.bigdata.optimaltimetobuy

/**
  * Created by sbellary on 6/12/2017.
  */

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}
import java.util.concurrent.TimeUnit
import com.cars.bigdata.optimaltimetobuy.RowMapper._
import com.typesafe.config.ConfigFactory
import play.api.libs.json._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.functions.udf
import org.apache.avro.Foo

object RandomForestRegression {

  case class VehicleDaily ( vehicle_id: String,
                            dma_code: String,
                            mileage: String,
                            price: String,
                            photo_count: String,
                            vdp_views: String,
                            total_vehicles: String,
                            median_price: String,
                            make_id: String,
                            make_model_id: String,
                            model_year_id: String,
                            trim_id: String)
  case class TrainingVehicleList(vehicle_id: String, days_infer:String, predicted_days: String, predicted_date: String)
  case class KeyValuePairTrain(otb:String, prediction:String)
  case class ActiveVehicleList(vehicle_id: String, predicted_days: String, predicted_date: String)
  case class KeyValuePair(otb:String, prediction:String)

  //DEFINE MAIN METHOD FOR GBT REGRESSION
  def main(args: Array[String]): Unit = {
    //SETUP LOGGING
//    Logger.getLogger("org").setLevel(Level.ERROR)
    System.setProperty("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")

    //SETUP CONFIG FACTOR FOR PRODUCTION DEPLOYMENT
    val appConf = ConfigFactory.load()
    val conf = new SparkConf()
      .setAppName("RandomForestRegression")
      .setMaster(appConf.getConfig(args(0)).getString("deploymentMaster"))
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.registerKryoClasses(Array(classOf[Foo]))

    //SETUP SPARK CONTEXT AND SQL CONTEXT
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //VALIDATE INPUT AND OUTPUT PATHS EXISTS
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val trainDataInputPathExists = fs.exists(new Path(appConf.getConfig(args(1)).getString("trainDataInputPath")))
    val rf_outputPathExists = fs.exists(new Path(appConf.getConfig(args(2)).getString("rf_outputPath")))
    val rfModelPathExists = fs.exists(new Path(appConf.getConfig(args(3)).getString("rf_modelpath")))
    val vehicleDailyInputPathExists = fs.exists(new Path(appConf.getConfig(args(4)).getString("vehicleDailyInputPath")))
    val rf_test_outputPathExists = fs.exists(new Path(appConf.getConfig(args(7)).getString("rf_test_outputPath")))

    //VALIDATE TRAIN DATA INPUT PATH
    if(!trainDataInputPathExists) {
      println("Invalid training data input path")
      return
    }

    //VALIDATE VEHICLE DAILY INPUT PATH
    if(!vehicleDailyInputPathExists) {
      println("Invalid vehicle daily input path")
      return
    }

    //VALIDATE AND OVERWRITE OUTPUT
    if(rf_outputPathExists)
      fs.delete(new Path(appConf.getConfig(args(2)).getString("rf_outputPath")), true)

    //VALIDATE AND OVERWRITE GBT MODEL OUTPUT
    if(rfModelPathExists)
      fs.delete(new Path(appConf.getConfig(args(3)).getString("rf_modelpath")), true)
    //VALIDATE
    if(rf_test_outputPathExists)
      fs.delete(new Path(appConf.getConfig(args(7)).getString("rf_test_outputPath")),true)

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
        |WHERE dma_code IS NOT NULL
        |AND mileage IS NOT NULL
        |AND price IS NOT NULL
        |AND photo_count IS NOT NULL
        |AND vdp_views IS NOT NULL
        |AND total_vehicles IS NOT NULL
        |AND make_id IS NOT NULL
        |AND make_model_id IS NOT NULL
        |AND model_year IS NOT NULL
        |AND trim_id IS NOT NULL
        |AND days >= 1 AND days <= 180
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
        |"20021O")
        |AND model_year IN ('2007O', '2008O', '2009O', '2010O', '2011O'
        |,'2012O','2013O','2014O','2015O','2016O','2017O')""".stripMargin)

    //DROP NULLS FROM THE DATA FRAME/FILL ZEROES FOR NULLS
    df = df.na.drop()
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
    df = df.na.drop()
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
    val rf = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("days_int")
      .setNumTrees(appConf.getConfig(args(5)).getInt("numTrees"))
      .setMaxDepth(appConf.getConfig(args(6)).getInt("depth"))
      .setMinInstancesPerNode(1)

    //TRAIN THE GBT MODEL WITH TRAINING DATA
    val rfModel = rf.fit(trainingData)

    //SAVE?PERSIST THE GBT MODEL
    sc.parallelize(Seq(rfModel)).saveAsObjectFile(appConf.getConfig(args(3)).getString("rf_modelpath"))

    //PREDICTIONS FOR TEST DATA
    var predictionsTest = rfModel.transform(testData)

    //SHOW OR SAVE RESULTS FOR TEST DATA
    predictionsTest = predictionsTest.select("vehicle_id", "days_int", "current_date", "prediction")
    predictionsTest = predictionsTest.withColumn("pred_days", predictionsTest("prediction").cast("Int"))
    predictionsTest.show

    val date_add = udf((x: String, y: Int) => {
      val sdf = new SimpleDateFormat("yyyy-MM-dd")
      val result = new Date(sdf.parse(x).getTime() + TimeUnit.DAYS.toMillis(y))
      sdf.format(result)
    } )
    import sqlContext.implicits._
    val dx = predictionsTest.select("vehicle_id", "days_int", "current_date", "pred_days")
    val dy = dx.withColumn("predicted_date", date_add($"current_date", $"pred_days"))
    dy.show()

    dy.registerTempTable("prediction_table")
    val testPredictionDF = sqlContext.sql("select * from prediction_table")

    val testPredictionRDD: RDD[Row] = testPredictionDF.rdd
    val testPredictionRow = testPredictionRDD.groupBy(rw => rw.getString(0)).map(f = row => {
      var testVehicleList = List[TrainingVehicleList]()
      val v_id = "\"otb::" + row._1+"\""
      val test_vehicle_id = row._2.foreach(row =>
        testVehicleList = TrainingVehicleList(row(0).toString, row(1).toString, row(3).toString,row(4).toString) :: testVehicleList
      )
      implicit val userFormat1 = Json.format[TrainingVehicleList]
      val jsonObj = Json.obj("prediction" -> testVehicleList)
      KeyValuePairTrain(v_id,jsonObj.toString())
      }
    )
    sqlContext.createDataFrame(testPredictionRow).registerTempTable("test_table")
    sqlContext.sql("select distinct * from test_table")
      .write.format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .option("quote", "\u0000")
      .save(appConf.getConfig(args(7)).getString("rf_test_outputPath"))

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

    //GET THE TIME TO TRAIN THE MODEL
    val end_time_train = Calendar.getInstance().getTime()
    val elapsed_time_train =  ((end_time_train.getTime() - start_time_train.getTime())/1000).toString
    println("Time taken to train the model: " + elapsed_time_train + " seconds.")

    //PREDICTIONS FOR TRAIN DATA DATA
    var predictionsTrain = rfModel.transform(trainingData)
    //EVALUATE RMSE FOR "TEST" DATA USING REGRESSION EVALUATOR
    val trainRMSE = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    //PRINT EVALUATION RESULTS FOR TRAIN DATA
    val rmse_train = trainRMSE.evaluate(predictionsTrain)
    println("Root Mean Squared Error (RMSE) on TRAIN data = " + rmse_train)

    //EVALUATE MAE FOR "TRAIN" DATA USING REGRESSION EVALUATOR
    val trainMAE = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    //PRINT EVALUATION RESULTS FOR TRAIN DATA
    val mae_train = testMAE.evaluate(predictionsTrain)
    println("Mean Absolute Error (MAE) on TRAIN data = " + mae_train)

    //EVALUATE R-SQUARED FOR "TRAIN" DATA USING REGRESSION EVALUATOR
    val trainRSquared = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    //PRINT EVALUATION RESULTS FOR TRAIN DATA
    val rsquared_train = trainRSquared.evaluate(predictionsTrain)
    println("R-Squared on Train data = " + rsquared_train)

    //EVALUATE MSE FOR "TRAIN" DATA USING REGRESSION EVALUATOR
    val trainMSE = new RegressionEvaluator()
      .setLabelCol("days_int")
      .setPredictionCol("prediction")
      .setMetricName("mse")

    //PRINT EVALUATION RESULTS FOR TEST DATA
    val mse_train = trainMSE.evaluate(predictionsTrain)
    println("Mean Squared Error (MSE) on Train data = " + mse_train)


    //************************************************************************************************
    //*****************************************PREDICT ACTIVE LISTINGS********************************
    //************************************************************************************************

    val vehicleDailyRaw1 = sqlContext.read.option("mergeSchema", "true").parquet(appConf.getConfig(args(4)).getString("vehicleDailyInputPath"))
    //val vehicleDailyRaw1 = sqlContext.read.format("csv").textFile(appConf.getConfig(args(4)).getString("vehicleDailyInputPath"))

    val vehicleDailyData = vehicleDailyRaw1.map(row => {
      val vehicle_id = row(0).toString
      val dma_code = row(2).toString
      val mileage = if (row.get(3) == null) null else row.getInt(3).toString
      val price = if (row.get(4) == null) null else row.getInt(4).toString
      val photo_count = row(5).toString
      val vdp_views = row(6).toString
      val total_vehicles = if (row.get(7) == null) null else row.getInt(7).toString
      val median_price = if (row.get(8) == null) null else row.getInt(8).toString
      val make_id = row(9).toString
      val make_model_id = row(10).toString
      val model_year_id = row(11).toString
      val trim_id = row(12).toString

      val vehicleDailyRow: VehicleDaily = VehicleDaily(vehicle_id, dma_code, mileage, price, photo_count, vdp_views, total_vehicles, median_price, make_id, make_model_id, model_year_id, trim_id)
      vehicleDailyRow
    })
    //val hqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
    val vehicleDFrame = sqlContext.createDataFrame(vehicleDailyData)
    val filteredDFrame = vehicleDFrame.na.fill(0)
    filteredDFrame.registerTempTable("vDF")
    filteredDFrame.show()

    //ADD A NEW COLUMN/FEATURE TO OUR DATA FRAME "PRICE_RATIO" AND CREATE MUTABLE DATA FRAME "DF"
    val vdf2 = sqlContext.sql(
      """SELECT vehicle_id,
        |dma_code,
        |mileage,
        |price,
        |photo_count,
        |vdp_views,
        |total_vehicles,
        |make_id,
        |make_model_id,
        |model_year_id,
        |trim_id,
        |current_date() AS current_date
        |FROM vDF
        |WHERE dma_code IS NOT NULL
        |AND mileage IS NOT NULL
        |AND price IS NOT NULL
        |AND photo_count IS NOT NULL
        |AND vdp_views IS NOT NULL
        |AND total_vehicles IS NOT NULL
        |AND make_id IS NOT NULL
        |AND make_model_id IS NOT NULL
        |AND model_year_id IS NOT NULL
        |AND trim_id IS NOT NULL
        |AND make_id IN ("20032O","20025O","20061O","20012O","20001O","20041O","20019O","20005O","20024O","20035O","20081O","20088O","20049O","20016O","20077O",
        |"20064O","20070O","20017O","20089O","20038O","20085O","20066O","20073O","20042O","20030O","20074O","2068O","20075O","20053O","44763O","20052O","20028O",
        |"20015O","20044O","20008O","20006O","20021O")
        |AND model_year_id IN ('2007O', '2008O', '2009O', '2010O', '2011O','2012O','2013O','2014O','2015O','2016O','2017O')""".stripMargin)
    vdf2.show()
    //DROP NULLS FROM THE DATA FRAME/FILL ZEROES FOR NULLS
    var vdf = vdf2.na.drop()
    vdf.show()

    //CREATE INDEXES AND ON-HOT ENCODED VECTORS FOR CATEGORICAL FEATURES
    //INDEX AND ENCODE DMA_CODE
    val v_dma_code_indexer = new StringIndexer().setInputCol("dma_code").setOutputCol("dma_code_IDX")
    vdf = v_dma_code_indexer.fit(vdf).transform(vdf)
    val v_dma_code_encoder = new OneHotEncoder().setInputCol("dma_code_IDX").setOutputCol("dma_sparse_vect")
    vdf = v_dma_code_encoder.transform(vdf)

    //INDEX AND ENCODE MAKE_ID
    val v_make_id_indexer = new StringIndexer().setInputCol("make_id").setOutputCol("makeIDX")
    vdf = v_make_id_indexer.fit(vdf).transform(vdf)
    val v_make_id_encoder = new OneHotEncoder().setInputCol("makeIDX").setOutputCol("make_sparse_vect")
    vdf = v_make_id_encoder.transform(vdf)

    //INDEX AND ENCODE MAKE_MODEL_ID
    val v_make_model_indexer = new StringIndexer().setInputCol("make_model_id").setOutputCol("make_model_id_IDX")
    vdf = v_make_model_indexer.fit(vdf).transform(vdf)
    val v_make_model_encoder = new OneHotEncoder().setInputCol("make_model_id_IDX").setOutputCol("make_model_sparse_vect")
    vdf = v_make_model_encoder.transform(vdf)

    //INDEX AND ENCODE MODEL_YEAR
    val v_model_year_indexer = new StringIndexer().setInputCol("model_year_id").setOutputCol("model_year_IDX")
    vdf = v_model_year_indexer.fit(vdf).transform(vdf)
    val v_model_year_encoder = new OneHotEncoder().setInputCol("model_year_IDX").setOutputCol("model_year_sparse_vect")
    vdf = v_model_year_encoder.transform(vdf)

    //INDEX AND ENCODE TRIM_ID
    val v_trim_id_indexer = new StringIndexer().setInputCol("trim_id").setOutputCol("trim_id_IDX")
    vdf = v_trim_id_indexer.fit(vdf).transform(vdf)
    val v_trim_id_encoder = new OneHotEncoder().setInputCol("trim_id_IDX").setOutputCol("trim_id_sparse_vect")
    vdf = v_trim_id_encoder.transform(vdf)

    //PERFORM DATA TYPE TRANSFORMATIONS FROM STRING TO DOUBLE
    vdf = vdf.withColumn("mileage_int", vdf("mileage").cast("double"))
    vdf = vdf.withColumn("price_int", vdf("price").cast("double"))
    vdf = vdf.withColumn("photo_count_int", vdf("photo_count").cast("double"))
    vdf = vdf.withColumn("vdp_views_int", vdf("vdp_views").cast("double")).cache()
    vdf = vdf.select("vehicle_id",
      "current_date",
      "dma_sparse_vect",
      "make_sparse_vect",
      "make_model_sparse_vect",
      "model_year_sparse_vect",
      "trim_id_sparse_vect",
      "mileage_int",
      "price_int",
      "photo_count_int",
      "vdp_views_int"
    )
    vdf = vdf.na.drop()
    vdf.show()

    //ASSEMBLE ALL OUR FEATURES AND CREATE "FEATURES" VECTOR USING "VECTOR ASSEMBLER"
    val vAssembler = new VectorAssembler()
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

    val vAssemblerData = vAssembler.transform(df)
    vAssemblerData.show()

    //PREDICTIONS FOR TEST DATA
    var predictActiveListings = rfModel.transform(vAssemblerData)

    //SHOW OR SAVE RESULTS FOR TEST DATA
    predictActiveListings = predictActiveListings.select("vehicle_id", "current_date", "prediction")
    predictActiveListings = predictActiveListings.withColumn("predicted_days", predictActiveListings("prediction").cast("Int"))
    predictActiveListings.show

    import sqlContext.implicits._
    val vdx = predictActiveListings.select("vehicle_id", "current_date", "predicted_days")
    val vdy = vdx.withColumn("predicted_buy_date", date_add($"current_date", $"predicted_days"))
    vdy.show()

    vdy.registerTempTable("prediction_table")
    val testPredDF = sqlContext.sql("select count(distinct vehicle_id) from prediction_table")
    testPredDF.show
    val predictionDF = sqlContext.sql("select * from prediction_table order by vehicle_id")

    val predictionRDD: RDD[Row] = predictionDF.rdd
    val predictionRow = predictionRDD.groupBy(rw => rw.getString(0)).map(f = row => {
      var vehicleList = List[ActiveVehicleList]()
      val vehicle_id = "\"otb::" + row._1+"\""
      val predict_vehicle_id = row._2.foreach(row =>
        vehicleList = ActiveVehicleList(row(0).toString, row(2).toString,row(3).toString) :: vehicleList
      )
      implicit val userFormat = Json.format[ActiveVehicleList]
      val vehic = Json.obj("prediction" -> vehicleList)
      KeyValuePair(vehicle_id,vehic.toString())
      }
    )
    val finalTable = sqlContext.createDataFrame(predictionRow).registerTempTable("finalTable")
    val testOutput = sqlContext.sql("select count(otb)  from finalTable")
    println(testOutput)

    val finalOutput = sqlContext.sql("select * from finalTable")
      finalOutput.write.format("com.databricks.spark.csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .option("quote", "\u0000")
      .save(appConf.getConfig(args(2)).getString("rf_outputPath"))

    //GET THE TOTAL TIME TO RUN THE PROGRAM
    val end_time_main = Calendar.getInstance().getTime()
    val elapsed_time_main =  ((end_time_main.getTime() - start_time_main.getTime())/1000).toString
    println("Total time to run the program : " + elapsed_time_main + " seconds.")

    //STOP SPARK CONTEXT
    sc.stop()
  }
}
