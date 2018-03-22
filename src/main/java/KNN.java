import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.*;

/**
 * Created by colntrev on 3/21/18.
 */
public class KNN {
    public static void main(String[] args){
        SparkSession session = SparkSession.builder().appName("KNN").getOrCreate();
        JavaSparkContext context = new JavaSparkContext(session.sparkContext());

        Integer k = Integer.parseInt(args[0]); // number of neighbors to consider
        Integer d = Integer.parseInt(args[1]); // number of dimensions in the data
        String training = args[2];
        String testing = args[3];

        // Broadcast objects maintain data that can be accessed across
        // nodes in the cluster
        final Broadcast<Integer> K = context.broadcast(k);
        final Broadcast<Integer> D = context.broadcast(d);

        JavaRDD<String> trainingSet = session.read().textFile(training).javaRDD();
        JavaRDD<String> testingSet = session.read().textFile(testing).javaRDD();
        JavaPairRDD<String, String> cartesianProduct = testingSet.cartesian(trainingSet);

        JavaPairRDD<String, Tuple2<Double,String>> knns = cartesianProduct.mapToPair((Tuple2<String, String> record)-> {
            String[] testRecord = record._1().split(";");
            String[] trainRecord = record._2().split(";");
            String testTemporaryID = testRecord[0];
            String testDimensions = testRecord[1];
            String classification = trainRecord[0];
            String trainDimensions = trainRecord[1];
            double distance = dist(testDimensions, trainDimensions, D.value());
            String key = testTemporaryID;
            Tuple2<Double, String> value = new Tuple2<>(distance,classification);
            return new Tuple2<>(key, value);
        });

        JavaPairRDD<String, Iterable<Tuple2<Double,String>>> knnByKey = knns.groupByKey();

        JavaPairRDD<String,String> output = knnByKey.mapValues((Iterable<Tuple2<Double,String>> neighbors) ->{
            SortedMap<Double, String> knn = findNearest(neighbors, K.value());
            Map<String, Integer> counts = countKs(knn);
            return majorityVote(counts);
        });

        output.saveAsTextFile("results");
        session.close();
    }

    public static double dist(String test, String train, Integer dims){
        List<Double> testList = stringToDoubleList(test);
        List<Double> trainList = stringToDoubleList(train);
        double sum = 0.0;
        for(int i = 0; i < dims; i++){
            double diff = trainList.get(i) - testList.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    public static List<Double> stringToDoubleList(String s){
        String[] tokens = s.split(",");
        List<Double> res = new ArrayList<>();
        for(String tok : tokens){
            res.add(Double.parseDouble(tok));
        }
        return res;
    }

    public static SortedMap<Double, String> findNearest(Iterable<Tuple2<Double,String>> n, Integer k){
        SortedMap<Double,String> res = new TreeMap<>();
        for(Tuple2<Double,String> neighbor : n){
            Double distance = neighbor._1();
            String classification = neighbor._2();
            res.put(distance, classification);
            if(res.size() > k){
                res.remove(res.lastKey());
            }
        }
        return res;
    }

    public static Map<String,Integer> countKs(SortedMap<Double,String> nearest){
        Map<String, Integer> map = new HashMap<>();
        for(Map.Entry<Double, String> entry : nearest.entrySet()){
            String classification = entry.getValue();
            Integer count = map.get(classification);
            if(count == null){
                map.put(classification, 1);
            } else {
                map.put(classification, count + 1);
            }
        }
        return map;
    }

    public static String majorityVote(Map<String, Integer> counts){
        String label = null;
        int votes = 0;
        for(Map.Entry<String,Integer> entry : counts.entrySet()){
            if(label == null){
                label = entry.getKey();
                votes = entry.getValue();
            } else {
                Integer count = entry.getValue();
                if(count > votes){
                    label = entry.getKey();
                    votes = count;
                }
            }
        }
        return label;
    }
}
