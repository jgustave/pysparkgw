name := "pysparkgw"

version := "0.1"

scalaVersion := "2.11.12"
val gcsVersion = "1.14.0"
val bqVersion = "0.32.0-beta"
val sparkVersion = "2.4.0"
val awsVersion = "1.11.564"


resolvers += Resolver.sbtPluginRepo("releases")
resolvers += Resolver.bintrayRepo("spark-packages","maven")
resolvers += Resolver.sonatypeRepo("public")
resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/devhome/projects/dproc/repo"

//mvn deploy:deploy-file -Durl=file:///Users/jerdavis/devhome/projects/dproc/repo/ -Dfile=/Users/jerdavis/devhome/projects/gdaxlog/target/gdaxlog-1.0-SNAPSHOT.jar -DgroupId=jd -DartifactId=gdaxlog -Dpackaging=jar -Dversion=1.0.8-SNAPSHOT
//libraryDependencies += "jd" % "gdaxlog" % "1.0.8-SNAPSHOT"

//libraryDependencies += "com.amazonaws" % "aws-java-sdk-s3" % awsVersion
//libraryDependencies += "com.amazonaws" % "aws-java-sdk-core" % awsVersion
//libraryDependencies += "com.amazonaws" % "aws-java-sdk-cloudwatch" % awsVersion

//libraryDependencies += "org.clapper" %% "grizzled-scala" % "4.9.2"
//libraryDependencies += "org.json4s" %% "json4s-native" % "3.4.2"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-hive" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

libraryDependencies += "ml.dmlc" % "xgboost4j-spark" % "0.90"
