#程序说明
###功能
本程序用于对发票进行图像分类，以api形式供外界调用
###分类流程
详见图表
![image][https://www.processon.com/view/link/5a290196e4b0f3a79867dd6b]

#model address
https://pan.baidu.com/s/1o7Nj1X4

#训练过程
###建立训练样本
   建立文件夹root，在root下建立与类别数相同的若干子文件夹，子文件夹名称为
类别名称，将对应类别的图片放入对应文件夹
###修改配置文件
   修改项目根目录下的trainConfigure.json，将"trainImageDir"后面的内容替换
为root的路径，如"/home/yqyi/root/",注意最后一个斜杠。
   按照新的类别名称修改其余内容
###启动训练
   启动src/train.py即可开始训练
###替换线上模型
   训练结果将默认存放在根目录下的logs文件夹，此文件夹为训练脚本建立，若存在，
   则会覆盖logs内同名的文件。训练结束后选取最后一次迭代产生的模型，替换掉根目录下
   weightsFile中的模型即可，名称必须一致。
###修改线上配置文件
   修改项目根目录下的predictConfigure.json，按照新的分类标准修改其中内容
