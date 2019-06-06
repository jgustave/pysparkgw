package jd
import ml.dmlc.xgboost4j.java.XGBoost
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{SparkSession, types}
import org.apache.spark.sql.types.{StructField, StructType}
//from pyspark.sql import DataFrame
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StringType
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import ml.dmlc.xgboost4j.scala.spark.XGBoost



/**
  *
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

    bar(1,"",sparkSession.sparkContext)
  }

      def quoteRandall = {
        println("Open unmarked doors")

      }

    def foo(a:Int, b:String) : Unit = {
      println("HelloWorld " + a + " " + b )
      import java.io._
      val pw = new PrintWriter(new File("/tmp/hello.txt" ))
      pw.write("HelloWorld " + a + " " + b)
      pw.close
    }

//  def bar(a:Int, b:String, jsc: JavaSparkContext) : String = {
//
//    val sc = JavaSparkContext.toSparkContext(jsc)
//
//    "HelloWorld " + a + " " + b + " " +sc.applicationId
//  }



  def bar(a:Int, b:String, jsc: JavaSparkContext) : String = {

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
    val df_raw = spark.read.option("header", "true").schema(schema).csv("/Users/jerdavis/devlib/xgboost/train.csv")
    val df = df_raw.na.fill(0)

    val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").setHandleInvalid("keep")

    val cabinIndexer = new StringIndexer().setInputCol("Cabin").setOutputCol("CabinIndex").setHandleInvalid("keep")

    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex").setHandleInvalid("keep")

    val vectorAssembler = new VectorAssembler().setInputCols( Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "CabinIndex", "EmbarkedIndex")).setOutputCol("features")

    val xgbParam = Map(
          "eta" -> 0.1f,
          "missing" -> 0.0,
          "objective" -> "binary:logistic",
          "num_round" -> 30,
          "maxBin" -> 256,
          "maxDepth" -> 10,
          "num_parallel_tree" -> 4)

    val xgbClassifier = new XGBoostClassifier(xgbParam).
          setFeaturesCol("features").
          setLabelCol("Survival").
          setPredictionCol("prediction")


    val pipeline = new Pipeline().setStages(Array(sexIndexer, cabinIndexer, embarkedIndexer, vectorAssembler, xgbClassifier))


    val dfs = df.randomSplit(Array(0.8, 0.2), seed=24)
    val trainDF = dfs(0)
    val testDF = dfs(1)


    trainDF.show(5)
    val model = pipeline.fit(trainDF)
    val result = model.transform(testDF).select( "PassengerId", "prediction")
    result.createOrReplaceTempView("rr")
    result.show()


    ""
  }



}
