from tensorflow.python.framework import graph_util
import tensorflow as tf
from tensorflow.contrib.layers import flatten
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
with tf.Graph().as_default() as graph:
    # ***** (1) Create Graph *****
    x = tf.placeholder(tf.float32, [1, 64, 64, 3])
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32],stddev=0.01))                
    b1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')+b1)   
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 


    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64],stddev=0.01))  
    b2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)       
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       



    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128],stddev=0.01))  
    b3 = tf.Variable(tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)       
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       



    W4 = tf.Variable(tf.random_normal([3, 3, 128, 256],stddev=0.01))  
    b4 = tf.Variable(tf.random_normal([256]))
    conv4 = tf.nn.relu(tf.nn.conv2d(pool3, W4, strides=[1, 1, 1, 1], padding='SAME') + b4)        
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')      


    Wf = tf.Variable(tf.random_normal([4*4*256,512],stddev=0.01))    
    bf = tf.Variable(tf.random_normal([512]))
    drop4_flat = tf.reshape(pool4, [-1, 4*4*256])        
    dense = tf.nn.relu(tf.matmul(drop4_flat, Wf) + bf)   


    Wout = tf.Variable(tf.random_normal([512,2],stddev=0.01))        
    bout = tf.Variable(tf.random_normal([2]))
    output = tf.add(tf.matmul(dense, Wout), bout, name="output")
    
    print('stats before freezing')
    stats_graph(graph)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ***** (2) freeze graph *****
        output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
        with tf.gfile.GFile('inference_graph.pb', "wb") as f:
            f.write(output_graph.SerializeToString())
# ***** (3) Load frozen graph *****
graph = load_pb('./inference_graph.pb')
print('stats after freezing')
stats_graph(graph)
