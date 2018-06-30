#import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS 
from pyspark.ml.evaluation import RegressionEvaluator
#import numpy
#from numpy import *
#from pyspark.sql import Row

conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationSystemWithSpark")
sc = SparkContext(conf=conf)

# getting the rating of the movie with user id and movie id
data = sc.textFile("file:///SparkCourse/ml-100k/u.data")

rdd = data.map(lambda x : x.split()).map(lambda x: int(x[0]),int(x[1]),float(x[2]))
features = ['user_id','movie_id','ratings']
df = rdd.toDF(features)

(train , test) = df.randomSplit([0.7,0.3])

als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="movie_id", ratingCol="rating",
          coldStartStrategy="drop")

model = als.fit(train)

prediction = model.transfrom(test)

value = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = value.evaluate(prediction)
print("Root-mean-square error = " + str(rmse))

userRecs = model.recommendForAllUsers(15)

movieRecs = model.recommendForAllItems(15)