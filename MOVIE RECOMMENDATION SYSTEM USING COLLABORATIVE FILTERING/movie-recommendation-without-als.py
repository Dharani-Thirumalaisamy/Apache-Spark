# importing the required libraries
from pyspark import SparkContext , SparkConf
import sys
from math import sqrt

# user defined function for movie names
def movienames():
    movie_list = {}
    with open ("ml-100k/u.ITEM", encoding='ascii', errors='ignore') as f:
        for lines in f :
            field = lines.split('|')
            movie_list[int(field[0])] = field[1]
    return movie_list
    
# user defined function for eliminating the duplicates
def duplicates(ratings):
    userratings = ratings[1]
    (movie1 , rating1) = userratings[0]
    (movie2 , rating2) = userratings[1]
    return movie1 != movie2
    
# user defined function for making movie as the key
def moviekey(movieratings):
    keypair = movieratings[1]
    (movie1 , rating1) = keypair[0]
    (movie2 , rating2) = keypair[1]
    return((movie1,movie2),(rating1,rating2))        
    
# calculating the similarity between the ratings using pearson correlation
def pearsoncorrelation(similarity):
    ratingspairs = 0
    sum_x  = 0
    sum_y  = 0
    sum_xy = 0
    sum_x2 = 0 
    sum_y2 = 0
    for variable_x , variable_y in similarity:
        sum_x  += variable_x
        sum_y  += variable_y
        sum_xy += variable_x * variable_y
        sum_x2 += variable_x * variable_x
        sum_y2 += variable_y * variable_y
        #ratingspairs += 1
        
    numerator = (ratingspairs * sum_xy) - ((sum_x)*(sum_y))
    denominator = sqrt((((ratingspairs*sum_x2)-(sum_x * sum_x))*((ratingspairs*sum_y2)-(sum_y * sum_y))))
    
    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, ratingspairs)
# Configuring the saprk context and distributing the work to all the cpu's in the system
conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationSystemWithSpark")
sc = SparkContext(conf=conf)

# loading the movie name from u.item
movie_names = movienames()

# getting the rating of the movie with user id and movie id
data = sc.textFile("file:///SparkCourse/ml-100k/u.data")

movie_ratings = data.map(lambda x:x.split()).map(lambda x: (int(x[0]),(int(x[1]),float(x[2]))))

# self-joining the data extracted to get the format of (userID , (movie1,rating1),(movie2,rating2))
self_join = movie_ratings.join(movie_ratings)

# filtering the duplicate entries
duplicates_filter = self_join.filter(duplicates)

# making the movie as key 
movie_key = duplicates_filter.map(moviekey)

# group by key - here , movie 
ratings_group = movie_key.groupByKey()

# Calculate the similarities between the ratings
ratings_similarities = ratings_group.mapValues(pearsoncorrelation).cache()

if (len(sys.argv) > 1):
    movie_id = int(sys.argv[1])
    
    score_threshold = 0.9
    watched_count = 75
    
    filteredResults = ratings_similarities.filter(lambda good_movie: \
        (good_movie[0][0] == movie_id or good_movie[0][1] == movie_id) \
        and good_movie[1][0] > score_threshold and good_movie[1][1] > watched_count)

    result = filteredResults.map(lambda good_movie : (good_movie[0],good_movie[1])).sortByKey(ascending = False).take(15)

    print("Top 15 similar movies for " + movie_names[movie_id])
    for results in result:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movie_id):
            similarMovieID = pair[1]
        print(movie_names[similarMovieID] + "\tscore: " + str(sim[0]) + "\tpeople watched: " + str(sim[1]))