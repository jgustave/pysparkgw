package jd
import java.util

import breeze.linalg.max
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StringType
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import scala.collection.JavaConverters._



/**
  * aws s3 cp ./target/scala-2.11/pysparkgw_2.11-0.1.jar s3://bitdatawest/pbin/
  */
object PyTest {
  def main(args : Array[String]) : Unit  = {
    println("Hello World")
    var sparkSession: SparkSession = null

    sparkSession = SparkSession.builder().appName("gendata").config("spark.ui.showConsoleProgress", true).getOrCreate()

    if (System.getProperties.getProperty("spark.master").contains("local")) {
      sparkSession.sparkContext.setCheckpointDir("/tmp/")
    }
    else {
      sparkSession.sparkContext.hadoopConfiguration.setInt("fs.s3.maxConnections", 5000)
      sparkSession.sparkContext.setCheckpointDir("s3://bitdatawest/data/chk/" + System.currentTimeMillis())
    }

    val features = Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "CabinIndex", "EmbarkedIndex")

    println("Best:"+ eval(sparkSession.sparkContext,
                          "/Users/jerdavis/devlib/xgboost/train.csv",
                          new util.ArrayList[String](features.toList.asJava),
                          "Survival",
                          .1f,
                          0f,
                          20,
                          100,
                          1f,
                          .9f,
                          1f,1f,1f,
                          1f,0f,
                          256,
                          30,
                          100,
                          10))

  }

  def test() : Unit = {
    println("################TEST############")
  }

  def eval (jsc: JavaSparkContext,
            path: String,
            features: util.ArrayList[String],
            response: String,
            eta: Double,
            gamma: Double,
            maxDepth: Int,
            maxLeaves: Int,
            minChildWeight: Double,

            //minChildWeight : Double,
            subsample : Double,
            sampleTree : Double,
            sampleLevel : Double,
            sampleNode : Double,
            lambda : Double,
            alpha : Double,
            maxBin: Int,
            numTrees: Int,
            numRounds: Int,
            numEarlyStoppingRounds: Int ) : Double = {
    println("HelloWorld Eval " + maxDepth)
    val sc = JavaSparkContext.toSparkContext(jsc)

    val schema = new StructType(
      Array(StructField("PassengerId", DoubleType),
        StructField("Survival", DoubleType),
        StructField("Pclass", DoubleType),
        StructField("Name", StringType),
        StructField("Sex", StringType),
        StructField("Age", DoubleType),
        StructField("SibSp", DoubleType),
        StructField("Parch", DoubleType),
        StructField("Ticket", StringType),
        StructField("Fare", DoubleType),
        StructField("Cabin", StringType),
        StructField("Embarked", StringType)
            ))

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()
    val df_raw = spark.read.option("header", "true").schema(schema).csv(path)
    val df = df_raw.na.drop()
    val dfs = df.randomSplit(Array(0.8, 0.2), seed=24)
    val trainDF = dfs(0)
    val testDF = dfs(1)

    val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").setHandleInvalid("keep")

    val cabinIndexer = new StringIndexer().setInputCol("Cabin").setOutputCol("CabinIndex").setHandleInvalid("keep")

    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex").setHandleInvalid("keep")

    //val vectorAssembler = new VectorAssembler().setInputCols( Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "CabinIndex", "EmbarkedIndex")).setOutputCol("features")

    val fa = features.toArray(Array(""))

    val vectorAssembler = new VectorAssembler().setInputCols( fa ).setOutputCol("features")

    val xgbParam = Map(
          "eta" -> eta,
          "gamma" -> gamma,

          "missing" -> 0.0,
          "objective" -> "binary:logistic",

          "max_depth" -> maxDepth,
          "min_child_weight" -> minChildWeight,

          "subsample" -> subsample,
          "colsample_bytree" -> sampleTree,
          "colsample_bylevel" -> sampleLevel,
          "colsample_bynode" -> sampleNode,

          "lambda" -> lambda,
          "alpha" -> alpha,

          "eval_metric" -> "aucpr",
            "tree_method" -> "hist",
            "grow_policy" -> "lossguide", //lossguide or depthwise
            "max_leaves" -> maxLeaves,

          "max_bin" -> maxBin,
          "num_parallel_tree" -> numTrees,

          "maximize_evaluation_metrics" -> "true",
          //"training_metric" -> "true",
          "num_round" -> numRounds,
          "num_early_stopping_rounds" -> numEarlyStoppingRounds,
          "timeout_request_workers" -> 60000L )

    val xgbClassifier = new XGBoostClassifier(xgbParam).
          setFeaturesCol("features").
          setLabelCol(response). //setLabelCol("Survival").
          setPredictionCol("prediction").
      setTrainTestRatio(.8)//.
          //setEvalSets(Map("evalSet"->testDF))


    val pipeline = new Pipeline().setStages(Array(sexIndexer, cabinIndexer, embarkedIndexer, vectorAssembler, xgbClassifier))


    val model = pipeline.fit(trainDF)

    val m =getModel(model.stages)
    //val tt =m.summary.trainObjectiveHistory
    val vv =m.summary.validationObjectiveHistory.toMap

    //val f = max(m.summary.trainObjectiveHistory)
    val best = max(vv("test"))

//    val result = model.transform(testDF).select( "PassengerId", "prediction")
//    result.createOrReplaceTempView("rr")
//    result.show()


    best
  }


  def evalFoo (jsc: JavaSparkContext,
            path: String,
            features: util.ArrayList[String],
            response: String,
            numWorkers: Int,
            eta: Double,
            gamma: Double,
            maxDepth: Int,
            maxLeaves: Int,
            minChildWeight: Double,
            subsample : Double,
            sampleTree : Double,
            sampleLevel : Double,
            sampleNode : Double,
            lambda : Double,
            alpha : Double,
            maxBin: Int,
            numTrees: Int,
            numRounds: Int,
            numEarlyStoppingRounds: Int ) : Double = {
    println("HelloWorld Eval " + maxDepth)
    val sc = JavaSparkContext.toSparkContext(jsc)

    val spark = SparkSession.builder.config(sc.getConf).getOrCreate()


    val df_raw = spark.read.parquet(path)
    val df = df_raw.na.drop().repartition(numWorkers)
    //val df = df_raw.na.drop()

//    val dfs = df.randomSplit(Array(0.8, 0.2))
//    val trainDF = dfs(0)
//    val testDF = dfs(1)

    val trainDF = df


    val fa = features.toArray(Array(""))

    val vectorAssembler = new VectorAssembler().setInputCols( fa ).setOutputCol("fvec")

    val scaler = new StandardScaler()
      .setInputCol("fvec")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    val xgbParam = Map(
      "num_workers" -> numWorkers,
      "nthread" -> 1,
          "eta" -> eta,
          "gamma" -> gamma,

          "missing" -> 0.0,
          "objective" -> "binary:logistic",

          "max_depth" -> maxDepth,
          "min_child_weight" -> minChildWeight,

          "subsample" -> subsample,
          "colsample_bytree" -> sampleTree,
          "colsample_bylevel" -> sampleLevel,
          "colsample_bynode" -> sampleNode,

          "lambda" -> lambda,
          "alpha" -> alpha,

          "eval_metric" -> "aucpr",
            "tree_method" -> "hist",
            "grow_policy" -> "lossguide", //lossguide or depthwise
            "max_leaves" -> maxLeaves,

          "max_bin" -> maxBin,
          "num_parallel_tree" -> numTrees,

          "maximize_evaluation_metrics" -> "true",
          //"training_metric" -> "true",
          "num_round" -> numRounds,
          "num_early_stopping_rounds" -> numEarlyStoppingRounds )

    val xgbClassifier = new XGBoostClassifier(xgbParam).
          setFeaturesCol("scaledFeatures").
          setLabelCol(response). //setLabelCol("Survival").
          setPredictionCol("prediction").
          //setEvalSets(Map("evalSet"->testDF)).
          setTrainTestRatio(.80)


    val pipeline = new Pipeline().setStages(Array(vectorAssembler, scaler, xgbClassifier))



    //trainDF.show(5)
    val model = pipeline.fit(trainDF)

    val m =getModel(model.stages)
    //val best = max(m.summary.trainObjectiveHistory)
    val validations = m.summary.validationObjectiveHistory.toMap
    val best = max(validations("test"))

    best
  }


  def getModel (stages: Array[Transformer]) : XGBoostClassificationModel= {
    stages.filter(i=>i.isInstanceOf[XGBoostClassificationModel])(0).asInstanceOf[XGBoostClassificationModel]
  }


}
