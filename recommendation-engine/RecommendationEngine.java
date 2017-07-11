
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;


public class RecommendationEngine {

	public static void main(String[] args) {
		
		// Turn off unnecessary logging
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		
		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("Recommendation Engine").setMaster("local[2]").set("spark.executor.memory","1g");;
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Read user-item rating file. format - userId,itemId,rating
		//JavaRDD<String> userItemRatingsFile = sc.textFile(args[0]);
		JavaRDD<String> userItemRatingsFile = sc.textFile("ratings.csv");
		System.out.println("ratings count " + userItemRatingsFile.count());
		
		// Read item description file. format - itemId, itemName, Other Fields,..
		//JavaRDD<String> itemDescritpionFile = sc.textFile(args[1]);
		JavaRDD<String> itemDescritpionFile = sc.textFile("games.csv");
		System.out.println("games count "+itemDescritpionFile.count());
		
		// Map file to Ratings(user,item,rating) tuples
		JavaRDD<Rating> ratings = userItemRatingsFile.map(new Function<String, Rating>() {
			public Rating call(String s) {
				String[] sarray = s.split(",");
				return new Rating(Integer.parseInt(sarray[0]), Integer
						.parseInt(sarray[1]), Double.parseDouble(sarray[2]));
			}
		});
		System.out.println("file to ratings map "+ ratings.count());
		
		// Create tuples(itemId,ItemDescription), will be used later to get names of item from itemId
		JavaPairRDD<Integer,String> itemDescritpion = itemDescritpionFile.mapToPair(
				new PairFunction<String, Integer, String>() {
			@Override
			public Tuple2<Integer, String> call(String t) throws Exception {
				String[] s = t.split(",");
				return new Tuple2<Integer,String>(Integer.parseInt(s[0]), s[1]);
			}
		});
		System.out.println("Item description tuples "+itemDescritpion.count());

		// Build the recommendation model using ALS
		
		int rank = 10; // 10 latent factors
		int numIterations = Integer.parseInt("10"); // number of iterations
		
		MatrixFactorizationModel model = ALS.trainImplicit(JavaRDD.toRDD(ratings),
				rank, numIterations);
		//ALS.trainImplicit(arg0, arg1, arg2)

		// Create user-item tuples from ratings
		JavaRDD<Tuple2<Object, Object>> userProducts = ratings
				.map(new Function<Rating, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(Rating r) {
						return new Tuple2<Object, Object>(r.user(), r.product());
					}
				});
		
		System.out.println("user-item tuples " + userProducts.count());
		
		// Calculate the itemIds not rated by a particular user, say user with userId = 1
		//// user id here is 0
		JavaRDD<Integer> notRatedByUser = userProducts.filter(new Function<Tuple2<Object,Object>, Boolean>() {
			@Override
			public Boolean call(Tuple2<Object, Object> v1) throws Exception {
				if (((Integer) v1._1).intValue() != 0) {
					return true;
				}
				return false;
			}
		}).map(new Function<Tuple2<Object,Object>, Integer>() {
			@Override
			public Integer call(Tuple2<Object, Object> v1) throws Exception {
				return (Integer) v1._2;
			}
		});
		
		System.out.println("not rated by  user "+ notRatedByUser.count());
		
		// Create user-item tuples for the items that are not rated by user, with user id 1
		JavaRDD<Tuple2<Object, Object>> itemsNotRatedByUser = notRatedByUser
				.map(new Function<Integer, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(Integer r) {
						return new Tuple2<Object, Object>(0, r);
					}
		});
		
		System.out.println("Items not rated by user "+ itemsNotRatedByUser.count());
		// Predict the ratings of the items not rated by user for the user
		JavaRDD<Rating> recomondations = model.predict(itemsNotRatedByUser.rdd()).toJavaRDD().distinct();
				
		// Sort the recommendations by rating in descending order
		recomondations = recomondations.sortBy(new Function<Rating,Double>(){
			@Override
			public Double call(Rating v1) throws Exception {
				return v1.rating();
			}
			
		}, false, 1);	
		
		System.out.println("total recommendations " + recomondations.count());
		
		// Get top 10 recommendations
		JavaRDD<Rating> topRecomondations = sc.parallelize(recomondations.take(10));
		
		System.out.println("top 10 recommendations "+ topRecomondations.count());
		
		// Join top 10 recommendations with item descriptions
		JavaRDD<Tuple2<Rating, String>> recommendedItems = topRecomondations.mapToPair(
				new PairFunction<Rating, Integer, Rating>() {
			@Override
			public Tuple2<Integer, Rating> call(Rating t) throws Exception {
				return new Tuple2<Integer,Rating>(t.product(),t);
			}
		}).join(itemDescritpion).values();
		
		System.out.println("recommended items "+ recommendedItems.count());
		//Print the top recommendations for user 1.
		recommendedItems.foreach(new VoidFunction<Tuple2<Rating,String>>() {
			@Override
			public void call(Tuple2<Rating, String> t) throws Exception {
				System.out.println(t._1.product() + "\t" + t._1.rating() + "\t" + t._2);
			}
		});
		
	}

}