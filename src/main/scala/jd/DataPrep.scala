package jd

import org.apache.spark.sql.functions.{col, isnull, lit}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * aws emr add-steps --cluster-id j-1BCCJULWS4JGJ --steps Type=Spark,Name=spec,Args=[--conf,spark.sql.windowExec.buffer.spill.threshold=4194000,--conf,spark.memory.fraction=0.95,--conf,spark.memory.storageFraction=0.05,--conf,spark.executor.extraJavaOptions="-Xss16m -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+UseParallelGC -XX:GCTimeRatio=99",--conf,spark.executor.memoryOverhead=20000,--conf,spark.executor.memory=200g,--conf,spark.executor.cores=30,--conf,spark.executor.instances=12,--conf,spark.dynamicAllocation.enabled=false,--class,jd.DataPrep,--deploy-mode,client,--master,yarn,--driver-memory,6g,s3://bitdatawest/pbin/pysparkgw_2.11-0.1.jar,s3://dataproc-data/data/gen/gen1,s3://dataproc-data/data/gen/gen1_1,u_300_2600_response,.0001],ActionOnFailure=CONTINUE
  */
object DataPrep {

  def main(args : Array[String]) : Unit  = {
    println("Hello World")

    val sparkSession = SparkSession.builder()
        .appName("tests")
        .config("spark.sql.windowExec.buffer.spill.threshold",4000000)
        .getOrCreate()

    import sparkSession.implicits._
    if(System.getProperties.getProperty("spark.master").contains("local")) {
      sparkSession.sparkContext.setCheckpointDir("/tmp/")
    }else {
      sparkSession.sparkContext.hadoopConfiguration.setInt("fs.s3.maxConnections", 5000)
      sparkSession.sparkContext.setCheckpointDir("s3://bitdatawest/data/chk/" + System.currentTimeMillis())
    }

    val input = args(0)
    val output = args(1)
    val response = args(2)
    val sample = args(3).toDouble

    println("input[" + input +"]")
    println("output[" + output +"]")
    println("response[" + response +"]")
    println("sample[" + sample +"]")


    println("loading..")
    var raw = sparkSession.read.parquet(input)
    balance(raw.na.drop(),response,false).sample(sample).write.parquet(output)
  }

  /**
    * Balance the classes by downsampling.
    * @param input
    * @param responseName
    * @param isDebug
    * @return
    */
  def balance (input: DataFrame, responseName:String, isDebug:Boolean ) :DataFrame = {

    var rCounts = input.groupBy(responseName).count()
    rCounts.show()
    rCounts = rCounts.filter(!isnull(col(responseName)))
    var countData = rCounts.orderBy(responseName).collect()
    val neg = countData(0).getLong(1)
    val pos = countData(1).getLong(1)

    if( pos < 10000 || neg < 10000 ) {
      println("Skipping " + responseName + " not enough examples ")
      //LOG.info("Skipping " + responseName + " not enough examples ")
      return input
    }

      val sampleRate = pos.toDouble/neg.toDouble

      if (sampleRate <= 1.0) {
        //LOG.info("{} Sample Rate {}", responseName, sampleRate)
        println(responseName + " Sample Rate " + sampleRate )
        val result = (input.filter(col(responseName) === lit(0.0)).sample(false, sampleRate)).union(input.filter(col(responseName) === lit(1.0)))
        if( isDebug ) {
          rCounts = result.groupBy(responseName).count()
          rCounts.show()
        }


        result
      }else {
        //LOG.info("{} Sample Rate {}", responseName, sampleRate)
        println(responseName + " Sample Rate " + sampleRate )
        val result =  (input.filter(col(responseName) === lit(1.0)).sample(false, 1.0/sampleRate)).union(input.filter(col(responseName) === lit(0.0)))
        if(isDebug) {
          rCounts = result.groupBy(responseName).count()
          rCounts.show()
        }
        result
      }
  }
}
