/**
 * Created by kalluri on 10/9/15.
 */

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.IDF

object TfIdf {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("TfIdf").setMaster("local[*]").set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)
    val documents: RDD[Seq[String]] = sc.textFile("/home/kalluri/IdeaProjects/Tf-Idf/src/main/resources/input").map(_.split(" ").toSeq)
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(documents)
    tf.cache()
    val idf = new IDF(minDocFreq = 1).fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
    println("working")
    tfidf.saveAsTextFile("/home/kalluri/IdeaProjects/Tf-Idf/src/main/resources/output")
    println("worked")
    sc.stop()
  }

}
