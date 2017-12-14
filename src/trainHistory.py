#-*-coding:utf-8-*-
import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/")
import core as model

def training(classNUM,imageW,imageH,
             batchSize,CAPACITY,maximum,
             learningRate,trainDIR,logDIR,bankNameList):
    imageList, labelList = getNameList(trainDIR,bankNameList)
    oneImageBatch, oneLabelBatch = makeBatch(imageList,labelList,
                                            imageW,imageH,
                                            batchSize,CAPACITY)
    forwardResult = model.yqyNet(oneImageBatch, batchSize, classNUM)
    loss = model.losses(forwardResult, oneLabelBatch)
    backwardOP = model.trainning(loss, learningRate)
    accuracy = model.evaluation(forwardResult, oneLabelBatch)


    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 分配50%
    sess = tf.Session(config=tf_config)


    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in range(maximum):
            if coord.should_stop():
                    break
            backwardResult, lossResult, accuracyResult = sess.run([backwardOP, loss, accuracy])
            if step % 5 == 0:
                print('Step %d,loss = %.2f,accuracy = %.2f%%' %(step, lossResult, accuracyResult*100.0))
            if step % 1000 == 0 or (step + 1) == maximum:
                checkpoint_path = os.path.join(logDIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

def getNameList(fileDir,bankNameList):
    bankImageNameList = []
    bankLabelNameList = []
    labelNUM= 0
    imageList=[]
    labelList=[]
    for bankName in bankNameList:
        bankImage=[]
        bankLabel=[]
        for file in os.listdir(fileDir + bankName):
            bankImage.append(fileDir + bankName+"/" + file)
            bankLabel.append(labelNUM)
        labelNUM=labelNUM+1
        print('There are %d %s' %(len(bankImage),bankName))

        bankImageNameList.append(bankImage)
        bankLabelNameList.append(bankLabel)

        imageList=np.hstack((imageList,bankImage))
        labelList=np.hstack((labelList,bankLabel))

    temp = np.array([imageList, labelList])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [round(float(i)) for i in label_list]
    return image_list, label_list

def makeBatch(imageList, labelList, imageW, imageH, batchSize, capacity):
    #transform format
    imageList = tf.cast(imageList, tf.string)
    labelList = tf.cast(labelList, tf.int32)
    nameQueue = tf.train.slice_input_producer([imageList, labelList])
    label = nameQueue[1]
    imageContent = tf.read_file(nameQueue[0])
    image = tf.image.decode_jpeg(imageContent, channels=3)
    image = tf.image.resize_images(image,
                                   [imageW, imageH],
                                   method=0)
    image = tf.image.per_image_standardization(image)
    oneImageBatch,oneLabelBatch = tf.train.shuffle_batch([image,label],
                                                      batch_size=batchSize,
                                                      num_threads=64,
                                                      capacity=capacity,
                                                      min_after_dequeue=capacity-1)
    oneLabelBatch = tf.reshape(oneLabelBatch,[batchSize])
    oneImageBatch = tf.cast(oneImageBatch,tf.float32)
    return oneImageBatch, oneLabelBatch

def firstCascade():
    logDIR =os.path.dirname(os.path.realpath(__file__))+"/../logs/first"
    trainImageDIR = os.path.dirname(os.path.realpath(__file__)) + '/../trainImage/first/'
    classNUM=7
    imageW=32
    imageH=32
    batchSize =10
    CAPACITY = 500
    maximum = 10000
    learningRate= 0.0001
    bankNameList = ["增值税发票", "定额发票", "火车票",
                    "银行回单", "长条发票", "飞机票",
                    "高速发票"]
    training(classNUM,imageW,
                     imageH,batchSize,
                     CAPACITY,maximum,
                     learningRate,trainImageDIR,
                     logDIR,bankNameList)

def secondCascade():
    logDIR = os.path.dirname(os.path.realpath(__file__)) + "/../logs/second"
    trainImageDIR = os.path.dirname(os.path.realpath(__file__)) + '/../trainImage/second/'
    classNUM=9
    imageW=32
    imageH=32
    batchSize =10
    CAPACITY = 500
    maximum = 10000
    learningRate= 0.0001
    bankNameList = ["上海农商银行1","中国银行1","兴业银行1",
                    "农业银行1", "华夏银行1", "工商银行1",
                    "建设银行1", "招商银行1", "民生银行1"]
    training(classNUM, imageW,
                     imageH, batchSize,
                     CAPACITY, maximum,
                     learningRate, trainImageDIR,
                     logDIR, bankNameList)

def continueTrain(classNUM,imageW,imageH,
             batchSize,CAPACITY,maximum,
             learningRate,trainDIR,logDIR,bankNameList):
   tf.reset_default_graph()
   importMeta=tf.train.import_meta_graph("****")




