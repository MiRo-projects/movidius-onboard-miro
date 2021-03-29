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
    W11 = tf.Variable(tf.random_normal([3, 3, 3, 32],stddev=0.01))          # kernel size(3,3)， Input channels(3)， Output channels(32)
    b11 = tf.Variable(tf.random_normal([32]))
    conv11 = tf.nn.relu(tf.nn.conv2d(x, W11, strides=[1, 1, 1, 1], padding='SAME')+b11)    # 64*64*32，Convolutional extraction of features, increasing the number of channels
    pool11 = tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32*32*64，Pooling, dimensionality reduction and complexity reduction


    Wf = tf.Variable(tf.random_normal([32*32*32,32],stddev=0.01))     # Input channels(32*32*32)， Output channels(32)
    bf = tf.Variable(tf.random_normal([32]))
    drop3_flat = tf.reshape(pool11, [-1, 32*32*32])         # -1 indicates that the row changes with the demand of the column，1*32768
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)      # [1,32768]*[32768,32]=[1,32]


    Wout = tf.Variable(tf.random_normal([32,2],stddev=0.01))        # Input channels(32)， Output channels(2)
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
