import sys 
import tensorflow as tf
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Config
#change based on knrm location
#sys.path.append('team2box/team2box/aaaidata/knrm/model/model_knrm.py')
sys.path.append('/home/dyx2/team2box')
sys.path.append('/home/dyx2/team2box/team2box/aaaidata/knrm/model')

from team2box.aaaidata.knrm.model import model_knrm
#import team2box.aaaidata.knrm.model.model_knrm
#import MYMODEL
import re,string 
#import nltk
#from nltk.corpus import stopwords
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只用 GPU:2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # 按需分配显存
    except RuntimeError as e:
        print(e)
"""
对每个数据集进行训练，构建了一个可以对“query（测试问题）”和“document（候选专家回答）”之间的语义匹配打分模型。
K-NRM 使用多个核函数来建模 query 和 document 的词级相似度分布，进行排序学习。
在训练阶段，使用提前处理好的问答配对数据进行监督训练；测试阶段，针对每一个 query 对所有候选文档打分，并返回 top-K 结果
"""
class myknrm:
  def __init__(self,path):
     #android: vocab size=10974 train=794 test=88 validation=37 qmaxlen=50 answermaxlen=100
     #data=["android","dba","physics","history"]# "mathoverflow",
     data=["android","dba","physics","mathoverflow","history"]
     Qmaxlen=[20,20,20,20,20]
     Dmaxlen=[100,100,100,100,100]
     Vocabsize=[10974,25182,31616,58321,34869]
     Trainsize=[794,2616,4421,9116,1528]
     Testsize=[88,290,491,1012,169]
     self.path = path
     self.config = Config()
     #data_ind=4
     #step 1: train
     # self.qmaxlen=Qmaxlen[data_ind]
     # self.dmaxlen=Dmaxlen[data_ind]
     # self.vocabsize= Vocabsize[data_ind]
     for j in range(5):
         self.qmaxlen = Qmaxlen[j]
         self.dmaxlen = Dmaxlen[j]
         self.vocabsize = Vocabsize[j]
         self.train(data[j], Trainsize[j],j+1)
     
     #step 2: test
     # self.path="/home/etemadir/QA/team2box/aaaidata/"+data[data_ind]
     # self.vocabsize= Vocabsize[data_ind]
     # self.testsize=Testsize[data_ind]
     # self.test2(50,data[data_ind])

  
  def train (self,data,train_size,i):
      #outdir=data+"/knrmformat/results/"
      outdir = self.path+ data +"/knrmformat/results/"
      if not os.path.exists(outdir):            
            os.mkdir(outdir) 
      #outdir=data+"/knrmformat/results/model"+str(i)+"/"
      outdir = self.path + data + "/knrmformat/results/model"+str(i)+"/"
      if not os.path.exists(outdir):            
            os.mkdir(outdir)       
      ob=model_knrm.Knrm(qmaxlen=self.qmaxlen,dmaxlen=self.dmaxlen,vocabsize=self.vocabsize,config=self.config)
      #ob.train(train_pair_file_path=data+"/knrmformat/train.txt", val_pair_file_path=data+"/knrmformat/validation.txt", train_size=train_size, checkpoint_dir=outdir)
      ob.train(train_pair_file_path=self.path + data + "/knrmformat/train_cleaned.txt",
               val_pair_file_path=self.path + data + "/knrmformat/validation_cleaned.txt",
               train_size=train_size, checkpoint_dir=outdir)

  def test2(self,TopK,data):
       ob=model_knrm.Knrm(20,100,self.vocabsize)
       testfile=open(self.path+"/team2box/testquestions.txt",encoding="utf-8")
       #results=open(self.path+"team2boxtestsets/results.txt","w", encoding="utf-8")
       path=self.path+ data + "/team2box/results.txt"
       allquerycode=[]
       qcode=testfile.readline().strip()
       while qcode:
          #print(qcode) 
          q=qcode.split(",")
          querycode=q[0].strip()
          allquerycode.append(querycode)
          qcode=testfile.readline().strip()
       testfile.close()  
       print(allquerycode) 
       # allscoresid, allscores=ob.test2(test_point_file_path=self.path+"/team2box/allquestions.txt"
       #                              , test_size=self.testsize
       #                              , output_file_path=self.path+"/team2box/output.txt"
       #                              ,load_model=True
       #                              ,checkpoint_dir=self.path+"/knrmformat/results/model10/"
       #                              , AllQuery=allquerycode, topk=TopK, resultpath=path)
       allscoresid, allscores = ob.test(test_point_file_path=self.path + data + "/team2box/allquestions.txt"
                                         , test_size=self.testsize
                                         , output_file_path=self.path + data + "/team2box/output.txt"
                                         , load_model=True
                                         , checkpoint_dir=self.path + "/knrmformat/results/model10/"
                                         , AllQuery=allquerycode, topk=TopK, resultpath=path)
              
          
       #for zz in range(len(allscoresid)):
        #     scoresid= allscoresid[zz]
       #      scores=allscores[zz]
       #      for i in range(TopK):
       #           results.write(str(scoresid[i])+" "+str(scores[i])+" ")
       #      results.write("\n")
        #     results.flush()
          
       
       #results.close()
     
  # def test3(self,  TopK):
  #      ob=MYMODEL.Knrm(20,100,self.vocabsize)
  #      testfile=open(self.path+"team2boxtestsets/testquestions.txt",encoding="utf-8")
  #      results=open(self.path+"team2boxtestsets/oursigirresults.txt","w", encoding="utf-8")
  #
  #      path=self.path+"team2boxtestsets/results.txt"
  #      allquerycode=[]
  #      qcode=testfile.readline().strip()
  #      while qcode:
  #         #print(qcode)
  #         q=qcode.split(",")
  #         querycode=q[0].strip()
  #         allquerycode.append(querycode)
  #         qcode=testfile.readline().strip()
  #      testfile.close()
  #      print(allquerycode)
  #
  #      scoresid, scores=ob.test2(test_point_file_path=self.path+"allposts.txt"
  #                                   , test_size=self.testsize
  #                                   , output_file_path=self.path+"team2boxtestsets/output.txt"
  #                                   ,load_model=True
  #                                   ,checkpoint_dir=self.path+"mymodelformat2/results/model6/"
  #                                   , AllQuery=allquerycode, topk=TopK,resultpath=path)
              
          #for i in range(TopK):
           #    results.write(str(scoresid[i])+" "+str(scores[i])+" ")
          #results.write("\n")
          #results.flush()
          #qcode=testfile.readline().strip()
       #testfile.close()
       #results.close()  
path = "/home/dyx2/team2box/team2box/data/"
ob=myknrm(path)
