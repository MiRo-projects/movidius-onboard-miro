from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_io
import numpy as np
import random
import cv2
import sys
import os

#Defining parameters
faces_my_path = './faces_my'
faces_other_path = './faces_other'
batch_size = 100          # Take 100 images at one time
learning_rate = 0.001     # learning rate
size = 64                 # picture size 64*64*3
imgs = []                 # Storing face images
labs = []                 # Store the labels corresponding to the face images

#Define a function to read face data
def readData(path , h = size , w = size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            #Enlarge the image to expand the edges of the image
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)                 # pictures of person's face added to the list of imgs
            labs.append(path)                # labels of person's face added to the list of labs

#Define the padding size
def getPaddingSize(img):
    height, width, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(height, width)

    if width < longest:
        tmp = longest - width
        left = tmp // 2
        right = tmp - left
    elif height < longest:
        tmp = longest - height
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right




#Define neural network layers, convolutional layer for feature extraction, pooling layer for dimensionality reduction, fully connected layer for classification
def cnnLayer():
    with tf.variable_scope('teacher',reuse = False) as sc :
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
        return output


def studentLayer():
    with tf.variable_scope('student',reuse = False) as sc :
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
        output = tf.add(tf.matmul(dense, Wout), bout, name="output")     # (1,32)*(32,2)=(1,2) ,compare with y_ [0,1]、[1,0] calculate loss
        return output


#define train function
def train():
    logit_teacher = cnnLayer()
    logit_student = studentLayer()
    
    logit_teacher_tem = logit_teacher / temperature
    logit_student_tem = logit_student / temperature

    cross_entropy_teacher = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_teacher, labels=y_))    
    cross_entropy1_student = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_student_tem, labels=tf.nn.softmax(logit_teacher_tem)))
    cross_entropy2_student = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_student, labels=y_))
    #soft and hard loss
    cross_entropy_student = 0.7 * cross_entropy1_student + 0.3 * cross_entropy2_student

    #Adam optimizer                                                           
    optimizer_teacher = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_teacher)
    optimizer_student = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_student)
    #calculate accuracy
    accuracy_student = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_student, 1), tf.argmax(y_, 1)), tf.float32))
    accuracy_teacher = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_teacher, 1), tf.argmax(y_, 1)), tf.float32))

    loss_teacher_summ = tf.summary.scalar('loss_teacher', cross_entropy_teacher)
    loss_teacher_summ = tf.summary.scalar('loss_student', cross_entropy_student)

    accuracy_teacher_summ = tf.summary.scalar('accuracy_teacher', accuracy_teacher)
    accuracy_student_summ = tf.summary.scalar('accuracy_student', accuracy_student)

    model_variables = tf.trainable_variables()
    var_teacher = [var for var in model_variables if 'teacher' in var.name]
    var_student = [var for var in model_variables if 'student' in var.name]

    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())
        saver1 = tf.train.Saver(var_teacher)
        saver2 = tf.train.Saver(var_student)
        pre_loss = 0
        for n in range(10):
            #100 pictures each time
            for i in range(num_batch):
                batch_x = train_x[i*batch_size: (i+1)*batch_size]          
                batch_y = train_y[i*batch_size: (i+1)*batch_size]          
                _, loss, summary = sess.run([optimizer_teacher, cross_entropy_teacher, merged_summary_op],
                                            feed_dict={x: batch_x, y_: batch_y})
                summary_writer.add_summary(summary, n*num_batch+i)
                print("step:%d,  loss:%g" % (n*num_batch+i, loss))
                if pre_loss-loss < 0.01 and n >= 2:
                    acc = accuracy_teacher.eval({x: test_x, y_: test_y})
                    print(acc)
                    saver1.save(sess, './teacher/train_faces.model', global_step=n*num_batch+i)
                    frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["teacher/output"])
                    graph_io.write_graph(frozen, './teacher', 'inference_graph.pb', as_text=False)
                    break
                pre_loss = loss
            if pre_loss-loss < 0.01 and n >= 2:
                break
            
        pre_loss = 0
        for n in range(10):
            for i in range(num_batch):
                batch_x = train_x[i*batch_size: (i+1)*batch_size]          
                batch_y = train_y[i*batch_size: (i+1)*batch_size]          
                _, loss, summary = sess.run([optimizer_student, cross_entropy_student, merged_summary_op],
                                            feed_dict={x: batch_x, y_: batch_y})
                summary_writer.add_summary(summary, n*num_batch+i)
                print("step:%d,  loss:%g" % (n*num_batch+i, loss))
                if pre_loss-loss < 0.01 and n >= 2:
                    acc = accuracy_student.eval({x: test_x, y_: test_y})
                    print(acc)
                    saver2.save(sess, './student/train_faces.model', global_step=n*num_batch+i)
                    frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["student/output"])
                    graph_io.write_graph(frozen, './student', 'inference_graph.pb', as_text=False)
                    sys.exit(0)
                pre_loss = loss

if __name__ == '__main__':


    readData(faces_my_path)
    readData(faces_other_path)
    imgs = np.array(imgs)                   # Converting image data and tags into arrays
    labs = np.array([[0, 1] if lab == faces_my_path else [1, 0] for lab in labs])  #Label: [0,1] means it's my face, [1,0] means other faces
    #Randomly divide the test set and the training set

    train_x_1, test_x_1, train_y, test_y = train_test_split(imgs, labs, test_size=0.1, random_state=random.randint(0, 100))
    train_x_2 = train_x_1.reshape(train_x_1.shape[0], size, size, 3)        
    test_x_2 = test_x_1.reshape(test_x_1.shape[0], size, size, 3)

    train_x = train_x_2.astype('float32')/255.0
    test_x = test_x_2.astype('float32')/255.0
    print('Train Size:%s, Test Size:%s' % (len(train_x), len(test_x)))

    num_batch = len(train_x) // batch_size
    x = tf.placeholder(tf.float32, [None, size, size, 3])                
    y_ = tf.placeholder(tf.float32, [None, 2])                            
    temperature = 10

    train()
