name := "optimal-time-to-buy"

version := "1.0"

organization := "com.cars"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.6.0" % "provided",
  "org.apache.spark" %% "spark-hive" % "1.6.0" % "provided",
  "org.scala-lang" % "scala-reflect" % "2.10.6" % "provided",
  "com.typesafe" % "config" % "1.3.1"
)

//logging
libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging-slf4j" % "2.1.2"

// testing
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

resolvers += "Typesafe Repo" at "http://repo.typesafe.com/typesafe/releases/"

libraryDependencies += "org.jboss.interceptor" % "jboss-interceptor-api" % "1.1"
resolvers += "JBoss" at "https://repository.jboss.org/"

scalaVersion := "2.10.6"

libraryDependencies += "play" % "play_2.10" % "2.1.0"

libraryDependencies += "com.databricks" % "spark-csv_2.10" % "1.5.0"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

        
