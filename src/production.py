#-*-coding:utf-8-*-
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/")
import predict as predictor

def parseJson(jsonDir):
    f = open(jsonDir, encoding='utf-8')
    content = json.load(f)
    return content

def predict(imagePath):
    jsonDir=os.path.dirname(os.path.realpath(__file__)) + "/../json/predictConfigure.json"
    logDir = os.path.dirname(os.path.realpath(__file__)) + "/../weightsFile"
    ckpt0 = "firstCascade"
    ckpt1 = "secondCascade"
    thresh0=0.7
    thresh1=0.7
    content=parseJson(jsonDir)
    list0=[]
    list1=[]
    for i in range(int(content['classNum0'])):
        list0.append(content['0'+str(i)])

    for i in range(int(content['classNum1'])):
        list1.append(content['1' + str(i)])

    resultFirst=predictor.evaluateOne(imagePath,
                                      logDir,
                                      int(content['classNum0']),
                                      ckpt0,
                                      thresh0)

    if resultFirst==1994 or resultFirst==3:
        resultSecond=predictor.evaluateOne(imagePath,
                                           logDir,
                                           int(content['classNum1']),
                                           ckpt1,
                                           thresh1)


        if resultSecond == 1994:
            return "银行回单"
        else:
            return list1[resultSecond]

    else:
        return list0[resultFirst]



