#coding=utf-8
import tensorflow as tf #导入tensorflow库
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np


def sigmoid_fun(x):
    return 1/(1+np.exp(-x))########################定义sigmoid函数将权重概率归一化



mnist = input_data.read_data_sets("MNIST_data", one_hot=True)#############################创建一个名字为mnis的文件夹
print("download ")
tf.reset_default_graph()
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])# NONE表示第一个维度可以是任意长度，mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

# Set model weights
W = tf.Variable(tf.random_normal([784, 87]))##################variable代表一个可修改的张量（变量），初始值随机数
b = tf.Variable(tf.zeros([87]))
# 构建模型: ax + b 
logit0 = tf.matmul(x, W) + b####################################      matmul(x,w)+b表示wx + b


# Set model weights
#W1 = tf.Variable(tf.random_normal([512, 128]))
#b1 = tf.Variable(tf.zeros([128]))
# 构建模型: ax + b 
#logit00 = tf.matmul(tf.nn.sigmoid(logit0), W1) + b1
#logit00 = tf.matmul(logit0, W1) + b1

# Set model weights
#W2 = tf.Variable(tf.random_normal([256, 128]))
#b2 = tf.Variable(tf.zeros([128]))
# 构建模型: ax + b 
#logit000 = tf.matmul(logit00, W2) + b2


# Set model weights
W3 = tf.Variable(tf.random_normal([87, 10]))
b3 = tf.Variable(tf.zeros([10]))
# 构建模型: ax + b 
logit = tf.matmul(tf.nn.sigmoid(logit0), W3) + b3########################tf.nn.sigmoid(w,x)w,x均需要事先定义才能使用






# Softmax分类
pred = tf.nn.softmax(logit) 
# Minimize error using cross entropy
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logit))###########################损失函数作差平方




#参数设置
learning_rate = 9.10#################################################两层隐藏层，学习率0.5之间迭代10，一层隐藏层学习率1.0迭代25次左右
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#####################################梯度下降函数（参数=学习率）
training_epochs = 35#####################################################每次训练都遍历完全部图片，遍历25次，epoch次数越多，准确性越高，准确性看loss
batch_size = 300#########################每次输入100张,数字越大，速度越快，准确度下降
display_step = 1#########################迭代1次输出结果
saver = tf.train.Saver()
model_path = "D:\programme\MNIST_data" # 模型保存路径（自定义）

#################################    训练    #######################################
# 启动session
with tf.Session() as sess:
    #saver.restore(sess, r"E:/mnist/a")
    ################################################加载上一次权重
    #####################################随机初始化用sess.run(tf.global_variables_initializer())# Initializing OP :初始化方法：Xavier#######使用上次的数据模型saver.restore(sess, r"E:/mnist/a")
    sess.run(tf.global_variables_initializer())# Initializing OP
   
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print( " Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

#################################    预测    #######################################

#读取模型
print("Starting 2nd session...")
with tf.Session() as sess:

    # 加载模型： 会话和模型路径
    saver.restore(sess, "D:\programme\MNIST_data")
    # ##########################如果没有模型使用随机初始化的方法
    #sess.run(tf.global_variables_initializer())
   #else:
       #sess.run(tf.global_variables_initializer())
     #Restore model weights from previously saved model
    #saver.restore(sess, model_path)
    
     # 测试 model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
   # 构建输入结构
    a = cv2.imread(r"F:/ASUS/Desktop/Densenet-Tensorflow-master/MNIST/b.png")#333333###################导入三维rgb图片，转化为数组
    
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    ####################以上函数三维图片灰度化，将a数组转成二维（a（数组），转成bgr灰度（后面的函数）），cv2colo函数颜色转化
    ainout = cv2.resize(ag, (28, 28))
                        #####################################resize()将二维的ag数组压缩转化为像素为28乘以28
    
    test_dict = {
           x: ainout.reshape(1, 784)
        }


                        ##################创建字典x构建的图节点，后面重构图数据，将其转化为一维（1，784）
    
    # 运行模型： logit；构建的图模型； test_dict: 输入数据的字典
    pred = sess.run(logit, test_dict)
    print("0123456789各数字的权重分别为：",pred)
    print("0123456789各数字的权重分别为：",tf.nn.softmax(pred))








 
import cv2


cv2.imshow('a', a)#######################################演示图像用imshow
cv2.waitKey()##############################################等待键盘响应
cv2.destroyAllWindows()##################################键盘有任何动作退出
