from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("AmountSpentOnShopping")
sc = SparkContext(conf = conf) 

def parseline(lines):
    field = lines.split(',')
    customer_id = int(field[0])
    amount_spent = float(field[2])
    return (customer_id , amount_spent)

line = sc.textFile("file:///SparkCourse/customer-orders.csv")
rdd  = line.map(parseline)
expenditure = rdd.mapValues(lambda x:(x,1)).reduceByKey(lambda x,y:(x[0]+y[0] , x[1]+y[1]))
total_exp = expenditure.mapValues(lambda x:x[0])
results = total_exp.collect()
for result in results :
    print(result)