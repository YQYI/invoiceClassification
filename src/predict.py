#-*-coding:utf-8-*-
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/")
from PIL import Image
import numpy as np
import tensorflow as tf
import core as model

def evaluateOne(imageDIR,logDIR,classNUM,CKPTPath,thresh):
   myGraph=tf.Graph()
   with myGraph.as_default():
       x = tf.placeholder(tf.float32, shape=[32, 32, 3])
       image = tf.cast(x, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 32, 32, 3])
       logit = model.yqyNet(image, 1, classNUM)
       logit = tf.nn.softmax(logit)
       saver = tf.train.Saver()

   sess = tf.Session(graph=myGraph,config=tf.ConfigProto(device_count={'cpu':0}))
   saver.restore(sess, logDIR + "/" + CKPTPath)
   imageArray = Image.open(imageDIR)
   imageArray = imageArray.resize([32, 32])
   imageArray = np.array(imageArray)
   prediction = sess.run(logit, feed_dict={x: imageArray})
   maxIndex = np.argmax(prediction)
   maxValue=np.max(prediction)
   if maxValue<thresh:
       return 1994
   else:
       return int(maxIndex)



