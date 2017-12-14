#-*-coding:utf-8-*-
import production as predictor
import os
count=0
imagePath="//home/yqyi/data/invoiceClassify/validateImage/all/"
imageNameList=os.listdir(imagePath)
for imageName in imageNameList:
    print(count)
    count=count+1
    result=predictor.predict(imagePath+imageName)
    os.rename(imagePath+imageName,
              imagePath+result+'-'+ str(count) +'.jpg')