#-*-coding:utf-8-*-
import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/")
import core as model
import json

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

def parseJson(jsonDir):
    f = open(jsonDir, encoding='utf-8')
    content = json.load(f)
    return content


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
    else:
        print(path + ' 目录已存在')


def runTrainning():
    content=parseJson(os.path.dirname(os.path.realpath(__file__)) + "/../json/trainConfigure.json")
    logDIR =os.path.dirname(os.path.realpath(__file__)) + "/../trainResult"
    mkdir(logDIR)
    trainImageDIR = str(content['trainImageDir'])
    classNUM=int(content['classNum'])

    imageW=32
    imageH=32
    batchSize =10
    CAPACITY = 500
    maximum = 10000
    learningRate= 0.0001
    bankNameList = []

    for i in range(classNUM):
        bankNameList.append(content[str(i)])
    print(bankNameList)

    training(classNUM, imageW,
                     imageH, batchSize,
                     CAPACITY, maximum,
                     learningRate, trainImageDIR,
                     logDIR, bankNameList)
runTrainning()



