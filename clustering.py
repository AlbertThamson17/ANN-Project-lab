import matplotlib.pyplot as plt
import tensorflow as tf 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA

def load_data() :
    df = pd.read_csv("D:\E202-COMP7117-TD01-00 - clustering.csv")
    feature = df[['ProductRelated_Duration','ExitRates','SpecialDay','VisitorType','Weekend']]
    target = df[['Revenue']]
    ordinal  = OrdinalEncoder()
    feature[['SpecialDay','VisitorType','Weekend']]=ordinal.fit_transform(feature[['SpecialDay','VisitorType','Weekend']])
    one_hot= OneHotEncoder(sparse = False)
    target = one_hot.fit_transform(target)

    feature = StandardScaler().fit_transform(feature)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(feature)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['data 1', 'data 2', 'data 3']).astype(object)

    
    return (principalDf)

dataset = load_data()
print(dataset.dtypes)
print (dataset)
dataset = dataset.to_numpy()
print (dataset)

layer = {
    'input': 3, 
    'hidden': 5,
    'output' : 1
}

weight = {
    'input-hidden' : tf.Variable(tf.random_normal([layer['input'],layer['hidden']])),
    'hidden-output' : tf.Variable(tf.random_normal([layer['hidden'],layer['output']]))
}

bias = {
    'input-hidden' : tf.Variable(tf.random_normal([layer['hidden']])),
    'hidden-output' : tf.Variable(tf.random_normal([layer['output']]))
}

epoch = 5000

class SOM:
    def __init__(self,height,width,input_dimension):
        self.height = height
        self.width = width
        self.input_dimension = input_dimension
        self.location = [tf.to_float([y,x]) for y in range(height) for x in range (width)]
        self.weight = tf.Variable(tf.random_normal([width*height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        self.weight = tf.Variable(tf.random_normal([width*height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        best_matching_unit = self.get_bmu()

        self.updated_weight, self.rate_stacked = self.update_neighbor(best_matching_unit)
        self.cluster  = None
    
    def get_bmu(self):
        square_difference = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1))

        bmu_index = tf.argmin(distance)
        bmu_location = tf.to_float([tf.div(bmu_index, self.width), tf.mod(bmu_index, self.width)])

        return bmu_location
    
    def update_neighbor(self, bmu):
        learning_rate = 0.1

        sigma = tf.to_float(tf.maximum(self.width, self.height)/2)

        square_difference = tf.square(self.location-bmu)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1))

        neighbor_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))
        rate = neighbor_strength * learning_rate

        total_node = self.width * self.height
        rate_stacked = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.input_dimension]) for i in range(total_node)])

        input_weight_difference = tf.subtract(self.input, self.weight)
        weight_difference = tf.multiply(rate_stacked, input_weight_difference)
        weight_new = tf.add(self.weight, weight_difference)

        return tf.assign(self.weight, weight_new), rate_stacked
    
    def train(self, dataset, epoch):
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            for i in range (epoch):
                for data in dataset:
                    sess.run(self.updated_weight, feed_dict = {self.input: data})
                    
            cluster = [[] for i in range(self.height)]
            location = sess.run(self.location)
            weight = sess.run(self.weight)
            
            for i, loc in enumerate(location):
                print(i,loc[0])
                cluster[int(loc[0])].append(weight[i])
                
            self.cluster = cluster
    
height = 3
width = 3
input_dimension = 3

som = SOM(height,width,input_dimension)
som.train(dataset,epoch)
plt.imshow(som.cluster)
plt.show()    