import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from matplotlib.patches import Rectangle
import networkx as nx
from tqdm import tqdm

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np 
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
"""
        读取图节点基本信息（问题、回答、专家数量）。
        加载图结构（从文件读取 CQAG.txt）。
        加载团队嵌入中心（teamsembeding.txt）和偏移（teamsOffsets.txt）。
        加载每个专家最匹配的团队（ExpertBestQuetionAnswer.txt）。
        从HCN中构造专家与问题的交互图，为“问题 + 专家”构建嵌入。
        通过随机游走生成节点序列，从中提取正负样本对。
        在训练过程中，将专家的初始嵌入限制在其对应团队的语义球体中
        初始化两个嵌入矩阵 W1 和 W2（前者部分嵌入初始化为团队中心附近的随机点）。
"""
class ExpertsEmbeding:        
    def __init__(self,hsize,data):
        """
        :param hsize:嵌入向量维度
        :param data:数据集路径

        """
        self.dataset=data
        #3self.load_graph()
        #self.save_qraph()
        #sys.exit()
        pfile=open(self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.N=int(properties[0]) # 图总节点数 CQA network graph N=|Qestions|+|Answers|+|Experts|
        self.qnum=int(properties[1]) #问题数量
        self.anum=int(properties[2]) #答案数量
        self.enum=int(properties[3]) #专家数量
        self.G={}  #初始化图结构变量 G，这是一个字典
        #G[node_id] = {'n': [邻居列表], 'w': [对应权重]}
        self.loadG()   #调用 loadG() 方法来加载 CQAG.txt 图结构文件
        #self.displayG()
        
        #加载团队的中心向量，每一行是一个团队的嵌入中心
        self.teamcenters=np.loadtxt(self.dataset+"/team2box/teamsembeding.txt")
        print(self.teamcenters)

        #加载团队“偏移量”，表示团队成员围绕中心的最大偏移半径（类似一个圆的半径或盒子大小）
        self.teamoffsets=np.loadtxt(self.dataset+"/team2box/teamsOffsets.txt")
        print(self.teamoffsets)

        # 读取每个专家的最佳团队和对应回答问题的分数
        #文件格式  ExpertID TeamID BestQuestionID Score
        gfile=open(self.dataset+"/ExpertBestQuetionAnswer.txt")        
        gfile.readline()
        line=gfile.readline().strip()
        self.ebt={} #存入 self.ebt 中
        #格式形如 self.ebt[expert_id] = [team_id, question_id, score]
        while line:
            ids=line.split(" ")
            self.ebt[int(ids[0])]=[int(ids[1]),int(ids[2]),float(ids[3])]
            line=gfile.readline()
        print("ebt=",self.ebt)
        gfile.close()
        
        self.hidden_size=hsize
        #初始化一个大小为 (问题数 + 专家数) × 嵌入维度 的嵌入矩阵 W1
        self.W1=ExpertsEmbeding.weight_variable((self.qnum+self.enum,self.hidden_size))  #问题与专家嵌入，问题部分随机初始化
        #专家部分的嵌入（W1[qnum:]）使用自定义逻辑初始化（ weight_variable_experts），使其靠近其所在团队的中心
        self.W1[self.qnum:self.qnum+self.enum].assign(self.weight_variable_experts((self.enum,self.hidden_size)))
        #初始化另一个嵌入矩阵 W2，将用于训练目标对比（例如 Skip-gram 中的上下文词）
        self.W2=ExpertsEmbeding.weight_variable((self.qnum+self.enum,self.hidden_size)) 
        #self.W1=self.weight_variable_experts((self.enum,self.hidden_size))
        #self.W2=ExpertsEmbeding.weight_variable((self.enum,self.hidden_size))
        #self.displayEmbedding()
    
    def weight_variable(shape):
        """
        创建一个 TensorFlow 可训练变量，用于初始化嵌入矩阵的权重
        :shape:一个二元组，矩阵维度
        :return: shape 维度的 TensorFlow权重张量
        """
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1]) #计算出的权重边界，避免梯度爆炸或消失
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp) #从范围 [-tmp, tmp] 中均匀采样初始化张量
        return tf.Variable(initial)
    
    def weight_variable_experts(self,shape): 
        """
        专家嵌入初始化
        :param shape:一个元组，形如 (self.enum, self.hidden_size)，表示要为 enum 个专家生成 hidden_size 维的嵌入。
        :return:返回一个 TensorFlow 可训练变量 tf.Variable，形状为 (enum, hidden_size)，即每个专家的初始嵌入
        确保专家初始位置在其最佳匹配团队的“圆形”区域中随机分布。
        利用极坐标随机生成一个偏移点（在圆内均匀分布），加到团队中心上。
        每个维度（2D中的X和Y）对应不同角度方向的偏移，模拟团队“盒子”的边界。
        """
        x=[] #保存所有专家的嵌入向量
        for i in range(self.enum):
            expertid=i+self.qnum+self.anum #得到专家在原图中的 ID（CQA图中专家是从 qnum+anum 开始）
            eteamid=self.ebt[expertid][0] #从 self.ebt 查找该专家所属的最佳团队编号
            offsets= self.teamoffsets[eteamid]  #获取该团队的偏移量
            #print(offsets)
            #在高维空间内模拟“极坐标随机点”
            r = offsets * np.sqrt(np.random.uniform(0,1)) #控制在圆内的距离（保留半径范围内均匀性）
            #print("r",r)
            theta = np.random.uniform(0,1) * 2 * 3.14 #方向
            
            x.append([]) #创建当前专家的嵌入向量
            #print("shape",shape[1])
            #print(self.teamcenters[self.ebt[expertid][0]])
            #遍历每个维度 j，生成嵌入：每两个维度看作一个 (x, y) 平面：一个用 cos，一个用 sin。这样多个维度组合成一个高维圆形区域。
            for j in range(shape[1]):
                if j%2==0:
                    #print("j=",j)
                    #self.teamcenters[eteamid][j]: 团队在第 j 维的中心坐标
                    x[i].append(self.teamcenters[self.ebt[expertid][0]][j]+ r[j] * np.cos(theta))
                else:
                    x[i].append( self.teamcenters[self.ebt[expertid][0]][j]+ r[j] * np.sin(theta))
        initial=np.array(x,dtype=np.float32)        
        return tf.Variable(initial)
    
    def loadG(self):  
        """
        HCN
        从数据集中读取图结构文件 CQAG.txt，将其转换为图的邻接表示形式 self.G，用于后续图嵌入训练或游走
        该图是一个无向加权图，节点表示问题、答案和专家，边表示它们之间的互动关系（如提问、回答等），权重表示交互强度
        G = {
            节点i: {
                    'n': [邻居节点编号列表],
                    'w': [对应边的权重列表]
                    }
             ...
            }
        """    
        gfile=open(self.dataset+"/CQAG.txt")
        #打开图结构文件，一般每行格式：节点1 节点2 权重
        e=gfile.readline() #读取当前行
        self.G={} #初始化图数据结构
        while e:
            #解析每一行，得到边的信息
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2]) #权重
            
            if i not in self.G:
                #如果 i 这个节点还没有记录，就为它创建邻接列表
                self.G[i]={'n':[],'w':[]}
            
            if j not in self.G:
                #为节点 j 初始化邻接信息
                        self.G[j]={'n':[],'w':[]}

            # 把节点 j 作为节点 i 的邻居，并记录权重
            self.G[i]['n'].append(j)
            self.G[i]['w'].append(w)
            #无向图，把 i 加入 j 的邻居列表
            self.G[j]['n'].append(i)
            self.G[j]['w'].append(w)
            e=gfile.readline()
        self.N=len(self.G)#设置节点总数
        #print(self.G)
        gfile.close()   
            
    def load_graph(self): 
        """从原始文本数据构建问答专家图HCN
        图的三类节点为：问题（Question），回答（Answer），专家（Expert）
        边表示：问题和回答之间的联系（问题有某些回答，带有评分），回答和专家之间的联系（回答是谁写的）
        最终结果是构建完整的图结构 self.G，用于嵌入训练。
        """
        self.G={} #初始化图结构
        qpfile=open(self.dataset+"/q_answer_ids_score.txt")
        qpfile.readline()# 跳过标题
        line=qpfile.readline().strip()
        qids=[]
        aids=[]
        eids=[]
        #提取所有问题和回答的编号（唯一性保证），并分别存入 qids, aids
        while line:
            qp=line.split(" ")
            qid=int(qp[0].strip())            
            if qid not in qids:
                qids.append(qid)
            caids=qp[1::2] 
            for aid in caids:
                if int(aid) not in aids:
                    aids.append(int(aid))
            line=qpfile.readline().strip()    
        qpfile.close()  
        print(len(qids))
        print(len(aids))
        #加载所有回答者 ID（专家）
        #pufile=open(self.dataset+"postusers.txt")
        pufile = open(self.dataset + "/answer_user_ids.txt")

        pufile.readline()
        line=pufile.readline().strip()
        #提取所有回答用户的 ID，作为专家（Experts）
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        print(len(eids))

        #统计各类节点数量
        self.qnum, self.anum, self.enum=len(qids), len(aids), len(eids)
        self.N=len(qids)+len(aids)+len(eids)
        
        #create CQA network graph
        #构建问题与回答之间的边 从 questionposts.txt 中加载每个问题的所有回答及其评分
        #qpfile=open(self.dataset+"/krnmdata1/questionposts.txt")
        qpfile = open(self.dataset + "/q_answer_ids_score.txt")
        qpfile.readline()
        line=qpfile.readline().strip()        
        while line:
            qp=line.split(" ")
            qid=qids.index(int(qp[0].strip()))           
            if qid not in self.G:
                self.G[qid]={'n':[],'w':[]}
                
            caids=qp[1::2] 
            #print(caids)
            caidsscore=qp[2::2] 
            #print(caidsscore)
            for ind in range(len(caids)):
                #每条边是 qid ↔ aid，双向加边，权重是该回答的得分
                aid=aids.index(int(caids[ind]))+len(qids)
                if aid not in self.G:
                    self.G[aid]={'n':[qid],'w':[int(caidsscore[ind])]}
                self.G[qid]['n'].append(aid)
                self.G[qid]['w'].append(int(caidsscore[ind]))
            line=qpfile.readline().strip()    
        qpfile.close()
        #读取每条回答是由哪个专家给出的
        #pufile=open(self.dataset+"/krnmdata1/postusers.txt")
        pufile = open(self.dataset + "/answer_user_ids.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            aid=aids.index(int(ids[0].strip()))+len(qids)
            eid=eids.index(int(ids[1].strip()))+len(qids)+len(aids)           
                      
            if eid not in self.G:
                self.G[eid]={'n':[aid],'w':[self.G[aid]['w'][0]]}
                
            else:
                self.G[eid]['n'].append(aid)
                self.G[eid]['w'].append(self.G[aid]['w'][0])
            self.G[aid]['n'].append(eid)
            self.G[aid]['w'].append(self.G[aid]['w'][0])    
            line=pufile.readline().strip()
            #建立回答（aid）与专家（eid）之间的双向边，使用原来回答的评分作为权重
        pufile.close()
    
        
    def save_qraph(self):
        """save CQA graph
        将已经构建好的 问答专家图 self.G 写入文件，分别保存为：
        邻接表形式的边列表（CQAG1.txt）
        节点统计信息（properties.txt）
        :return:
        写入 CQAG1.txt：图的边（无向、带权）；
        写入 properties.txt：图的结构参数。
        """
        qfile=open(self.dataset+"/CQAG1.txt","w")
        #qfile.write("N="+str(self.N)+" Questions= "+str(self.qnum)+" index=0.."+str(self.qnum-1)
        #            +"; Answers= "+str(self.anum)+" index="+str(self.qnum)+".."+str(self.qnum+self.anum-1)
        #            +"; Experts= "+str(self.enum)+" index="+str(self.qnum+self.anum)+".."+str(self.qnum+self.anum+self.enum-1)+"\n")

        #对于图中每个节点 node：
        # 遍历它的邻接点 self.G[node]['n'];为防止重复写边（因为是无向图），只写 node < neighbor 的情况
        # 将边 node neighbor weight 写入文件，每行一条
        for node in self.G:
            for i in range(len(self.G[node]['n'])):
                if node< self.G[node]['n'][i]:
                    qfile.write(str(node)+" "+str(self.G[node]['n'][i])+" "+str(self.G[node]['w'][i])+"\n")

        qfile.close()
        pfile=open(self.dataset+"properties.txt","w")
        #写出总节点数 N 和三类节点的编号范围
        # e.g.N=12000 Questions= 4000 index=0..3999; Answers= 4000 index=4000..7999; Experts= 4000 index=8000..11999
        pfile.write("N="+str(self.N)+" Questions= "+str(self.qnum)+" index=0.."+str(self.qnum-1)
                    +"; Answers= "+str(self.anum)+" index="+str(self.qnum)+".."+str(self.qnum+self.anum-1)
                    +"; Experts= "+str(self.enum)+" index="+str(self.qnum+self.anum)+".."+str(self.qnum+self.anum+self.enum-1)+"\n")
        pfile.write(str(self.N)+" "+str(self.qnum)+" "+str(self.anum)+" "+str(self.enum))
        pfile.close()
    
    def displayG(self):
        """
        将图结构 CQAG.txt 可视化。
            节点：所有问题、答案、专家
            边：带权连接，表示图中问答关系
        显示节点标签和边的权重
        :return: 图像窗口（Matplotlib 显示网络图）
        """
        G=nx.Graph()
        G=nx.read_weighted_edgelist(self.dataset+"/CQAG.txt")

        #创建边列表 edges，只保留权重大于0的边（虽然通常所有权都应为正）
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
        #计算布局坐标。使用 spring_layout 计算节点在 2D 平面的坐标。；k=0.2 是弹簧力参数（越大节点距离越远）；iterations=50 表示布局算法的迭代次数
        pos = nx.spring_layout(G,k=0.2,iterations=50)  # positions for all nodes
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=100)#,node_color='g',fill_color='w')
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)
        # labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        
        plt.axis('off')
        plt.show()
        
    def displayEmbedding(self):
        """
        绘制团队嵌入的中心点（teamcenters）
        以圆形表示团队的偏移半径（teamoffsets）
        绘制问题节点的嵌入位置（self.W1 中前 qnum 个）
        绘制专家节点的嵌入位置（self.W1 中从 qnum 起的 enum 个）
        图像中不同的颜色/形状表示不同类型的节点
        :return:Matplotlib 图像窗口（显示嵌入）
        """
        Centers=self.teamcenters.copy()
        Offsets=self.teamoffsets.copy()
        #print(embed)
        #model = TSNE(n_components=2, random_state=0)
        #y=model.fit_transform(embed) 
        y=Centers
        max1=np.amax(y,axis=0)+1.1*np.amax(Offsets,axis=0)
        min1=np.amin(y,axis=0)-1.1*np.amax(Offsets,axis=0)
        
        plt.figure(figsize=(5,5))
        plt.plot(y[0:len(Centers),0],y[0:len(Centers),1],'gx')
        
        for i in range(len(Centers)):
            plt.text(y[i,0],y[i,1], "t"+str(i), fontsize=8)  
        
        ax = plt.gca()    
        ax.set_aspect(1)
        for i in range(len(Centers)):
            #c= Rectangle((y[i,0]-Offsets[i,0],y[i,1]-Offsets[i,0]), 2*Offsets[i,0],2*Offsets[i,0] , linewidth=1,edgecolor='r',facecolor='none')
            c=plt.Circle((y[i,0], y[i,1]), Offsets[i,0], color='b', clip_on=False,fill=False)
            ax.add_patch(c)
        
        
        plt.plot(self.W1.numpy()[:self.qnum,0],self.W1.numpy()[:self.qnum,1],'b+',markersize=14)
        
        for i in range(self.qnum):
            plt.text(self.W1.numpy()[i,0],self.W1.numpy()[i,1], "q"+str(i), fontsize=8)
        
        plt.plot(self.W1.numpy()[self.qnum:,0],self.W1.numpy()[self.qnum:,1],'r*',markersize=12)
        
        for i in range(self.enum):
            plt.text(self.W1.numpy()[self.qnum+i,0],self.W1.numpy()[self.qnum+i,1], "e"+str(self.qnum+i), fontsize=8)
        #ax.set_xlim([min1[0], max1[0]])
        #ax.set_ylim([min1[1], max1[1]])       
        plt.show()
    
    
    def walker(self,start, walklen):
        """
        实现自定义的随机游走逻辑，避免回头（避免选中上一个节点）
        从指定节点 start 开始,按图中的邻接关系走 walklen 步,避免重复回到上一个节点（防止“抖动”）
        只记录“问题节点”和“专家节点”的访问轨迹（跳过回答节点）
        :param start:起始节点的编号
        :param walklen:随机游走的步数
        :return:一个游走路径字符串，节点编号以空格分隔。
        """
        walk=""
        ii=0        
        #start=random.randint(self.qnum+self.anum,self.N) # start from expert
        prev=start
        while ii<walklen: 
            #print("st="+ str(start)+" pre="+str(prev))            
            ind=0
            #如果当前节点（start）只有一个邻接节点，则必然跳过去，不做随机选择
            if len(self.G[start]['n'])==1:
                neib=self.G[start]['n']
                #print(neib)
                ind=0
            #多邻居时的随机跳转逻辑,复制邻居及其权重，避免破坏原图结构
            else:
                weights=self.G[start]['w'].copy()  
                neib=self.G[start]['n'].copy()
                #print(neib)
                #print(weights)
                #如果上一个节点是当前节点的邻居，去掉它（避免回头）
                if prev in neib:
                    indpre=neib.index(prev)                
                    del weights[indpre:indpre+1]
                    del neib[indpre:indpre+1]
                    #print(neib)
                    #print(weights)
                #使用“加权采样”选择一个邻接节点，跳转到它,权重高的边更容易被选中
                if len(neib)==1:
                    ind=0
                else:    
                    sumw=sum(weights)                
                    ranw=random.randint(1,sumw)
                    #print("sumw="+str(sumw)+" ranw="+str(ranw))                
                    for i in range(len(neib)):
                        if ranw<=sum(weights[0:i+1]):
                            ind=i
                            break
            #节点记录判断：只保留问题和专家的编号.对专家节点编号做偏移修正（减去 answer 数量）使得专家编号紧跟问题编号
            if start<self.qnum or start>self.qnum+self.anum:
                if start>self.qnum+self.anum:
                    start=start-(self.anum)
                walk+= " "+str(start )           
            prev=start
            start=neib[ind]
            
            #if start>self.qnum+self.anum:
            ii+=1
        return walk.strip()    
    
    def walks1(self,walklen,n):
        """
        批量调用 walker 方法，对所有问题节点和专家节点执行多次随机游走，并将结果存储为“游走路径序列”。
        此函数生成的数据类似于自然语言中的“句子”，每条路径是一个“词序列”，用于后续嵌入学习。
        :param walklen:每条游走的步数
        :param n:每个节点发起的游走次数
        :return:每条路径是一个字符串数组（节点编号）
        """
        data=[] #用于存放所有生成的游走路径
        #对所有问题节点执行 n 次游走，每次长度为 3
        for i in range(self.qnum):
            for j in range(n):
                walk=self.walker(i,3).split(" ")
                data.append(walk)
        #对所有专家节点执行 n 次游走，每次长度为 walklen
        for i in range(self.enum):
            for j in range(n):
                walk=self.walker(self.qnum+self.anum+i,walklen).split(" ")
                data.append(walk)
        return data
   
    def get_train_pair(self,walks,windowsize, N):
        """
        从“随机游走”序列中提取训练样本对（中心词与上下文词），同时生成用于负采样的负样本池
        :param walks:从 walks1() 得到的所有游走序列，每条是字符串列表
        :param windowsize:滑动窗口大小，控制上下文范围
        :param N:总节点数（初始化负采样表）
        :return:pairs	np.ndarray	正样本对 (中心词, 上下文词)，形状为 (样本数, 2)
                negs	np.ndarray	负采样用的节点 ID 池
        """
        #print(N)
        #初始化节点出现频率统计
        z=np.zeros((N)) #记录节点 i 在所有路径中出现的次数
        total=0  #是节点出现的总次数
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        #计算每个节点的出现概率及采样概率
        z1=z/total  #每个节点的频率
        z1[z1 == 0] = 1e-8  # 避免除以 0

        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #用于下采样高频节点的保留概率,高频词的 p 趋近于 0，低频词 p 趋近于 1
        #print(p)
        #构建负采样池
        z2=np.power(z,.75)  #使用 z^0.75 是 Word2Vec 中构建负样本概率分布的标准做法（经验公式）
        p2=z2/np.sum(z2)  #每个节点作为负样本的概率
        #print(p2)
        #每个节点按其负采样概率 p2[i]，将其复制多次进入 negsamples
        negsamples=[]
        for i in range(N):
            rep=int(p2[i]*100) 
            if rep==0:
               rep=1 
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)#打乱顺序，便于后续负样本随机挑选
        #print(negs)
        pairs=[]
        #遍历所有路径中的每一个“中心节点”
        for i in range(len(walks)):
            walk=walks[i]                     
            for context_ind in range(len(walk)):        #滑动窗口从 start 到 context_ind-1
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):  #使用概率 p 对当前中心词进行保留采样
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        if (100-p[int(walk[context_ind])]*100)>x:  #若随机数 x 超出保留阈值，跳过本样本（即下采样高频词）
                            continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])]) #将中心词和左侧上下文词组成一对加入 pairs
                        if i+context_ind+1<len(walk):
                            x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                            if (100-p[int(walk[context_ind])]*100)>x:
                                continue
                            if  walk[context_ind]!=walk[i+context_ind+1]:   
                                pairs.append([int(walk[context_ind]),int(walk[i+context_ind+1])]) #将右侧上下文词加入 pairs（右侧需要 +1）
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    
    def get_train_pair2(self,walks,windowsize, N):
        """
        从随机游走路径中生成：
            正样本对（中心词，上下文词）
            负样本池（用于负采样）
        不同于 get_train_pair()，这个版本去除了：
        中心词概率下采样（即不使用 p)
        它适用于不考虑节点频率对训练样本的影响的情况，更快，但可能精度略低。
        :param walks:
        :param windowsize:
        :param N:
        :return:
        """
        #print(N)
        z=np.zeros((N))
        total=0
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        z1=z/total
        z1[z1 == 0] = 1e-8  # 避免除以 0
        p = (np.sqrt(z1 / 0.001) + 1) * (0.001 / z1)

        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)
        #print(p2)
        negsamples=[]
        for i in range(N):
            rep=int(p2[i]*100)  
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        for i in range(len(walks)):
            walk=walks[i]                     
            for context_ind in range(len(walk)):            
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        #if (100-p[int(walk[context_ind])]*100)>x:
                        #    continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])])
                        if i+context_ind+1<len(walk):
                            #x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                            #if (100-p[int(walk[context_ind])]*100)>x:
                            #    continue
                            if  walk[context_ind]!=walk[i+context_ind+1]:   
                                pairs.append([int(walk[context_ind]),int(walk[i+context_ind+1])])
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    
    def walks(self,walklen,n1):
        """
        在 HCN 图上执行带权随机游走（biased random walks），使用 StellarGraph 的 BiasedRandomWalk 类
        游走结果用于训练 Word2Vec-style 嵌入
        并对结果进行预处理，去掉回答节点（answer nodes），只保留问题节点和专家节点。
        :param walklen:
        :param n1:
        :return:
        """
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/CQAG.txt")
        
        rw = BiasedRandomWalk(StellarGraph(G)) #使用 BiasedRandomWalk 类初始化游走生成器
        #执行有偏随机游走
        weighted_walks = rw.run(
        nodes=G.nodes(), # root nodes
        length=walklen,    # maximum length of a random walk
        n=n1,          # number of random walks per root node 
        p=0.5,         # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,         # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True, #for weighted random walks
        seed=42        # random seed fixed for reproducibility
        )
        print("Number of random walks: {}".format(len(weighted_walks)))
        #print(weighted_walks[0:10])
        
        #去除回答节点，只保留问题和专家节点
        walks=[]
        for i in range(len(weighted_walks)):
            walk=weighted_walks[i]
            w=[]
            for node in walk:
                if int(node)<self.qnum:
                    w.append(node)
                elif int(node)>(self.qnum+self.anum):
                    n=int(node)-self.anum
                    w.append(str(n))
            walks.append(w)        
        print(walks[0:10])
        #如果是问题节点（索引 < self.qnum），直接保留。
        # 如果是专家节点（索引 > self.qnum + self.anum）：
        # 减去 self.anum，因为要统一映射专家到 [qnum, qnum+enum) 区间（对应训练用的专家嵌入索引）并转回字符串加入结果。
        # 回答节点（索引 ∈ [qnum, qnum+anum)）被 跳过，不参与训练。
        return walks
    
    def loss(predicted_y, target_y):        
        return tf.reduce_mean(tf.square(predicted_y - target_y))

    def model(self,inputs_i,inputs_j):
        """
        神经网络前向传播过程
        :param inputs_i:被训练的主节点（例如 expert）索引列表
        :param inputs_j:与之配对的上下文节点（例如问题或负样本）索引列表
        :return:返回形状为 [len(inputs_j), 1] 的相似度分数矩阵 o，每行表示 inputs_i 中的节点与对应 inputs_j 的相似度（通过 sigmoid 映射后）
        """
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W2, inputs_j, name='jemb')          
        # Learning-To-Rank layer. o is the final matching score.
        temp=tf.transpose(j_embed, perm=[1, 0]) #对 j_embed 做转置：变成 (hidden_dim, batch_size)
        o = tf.sigmoid(tf.matmul(i_embed, temp)) #做内积计算：得到节点之间的相似度（余弦方向）,用 sigmoid 映射到 (0, 1)，表示概率
        o = tf.reshape(o, (len(o[0]),1))  #将最终输出调整为形状 [batch_size, 1]，便于与真实标签对比计算 loss
        #print("o=")
        #print(o)
        return o
    
    def train(self, inputs_i, inputs_j, outputs, learning_rate):
        """
        执行前向传播 + 反向传播 + 参数更新
        :param inputs_i:主节点索引（如专家）
        :param inputs_j:与其配对的上下文节点索引（正负样本）
        :param outputs:对应的标签（正样本为 1，负样本为 0）
        :param learning_rate:学习率
        :return:当前这一步的 loss（损失值）
        """
        #print(inputs_i)
        #获取嵌入向量
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W2, inputs_j, name='jemb')
        #使用 GradientTape 自动求导；调用 model() 得到预测分数 predicted_y；再调用 loss() 得到均方误差（MSE）
        with tf.GradientTape() as t:
            current_loss = ExpertsEmbeding.loss(self.model(inputs_i,inputs_j), outputs)
        #计算损失函数对两个嵌入矩阵的梯度
        dW1, dW2 = t.gradient(current_loss, [self.W1, self.W2])
        #对 inputs_i 这些位置的 W1 进行梯度下降更新
        i_embed=i_embed-(learning_rate * dW1.values)
                
        k1=0
        #print(inputs_i.numpy())
        #对问题节点（question）直接更新
        # 对专家节点（expert）：检查更新后是否仍在所属团队的“圆形区域”内。如果在，就更新；否则保持不变（保持专家在原团队语义边界内）
        for k in inputs_i.numpy():
            #print(k)
            if k<self.qnum:
                self.W1[k,:].assign(i_embed[k1,:])
            else:
                teamcenter=np.array(self.teamcenters[self.ebt[self.anum+k][0]])
                c=tf.square(tf.subtract(i_embed,teamcenter))         
                d=tf.sqrt(tf.reduce_sum(c,axis=1)).numpy()[0]
                #print("d=",d)
                if d<self.teamoffsets[self.ebt[self.anum+k][0]][0]:
                    #print("offset=",self.teamoffsets[self.ebt[self.anum+k][0]][0])
                    self.W1[k,:].assign(i_embed[k1,:])
            k1+=1
        
        j_embed=j_embed-(learning_rate * dW2.values) #更新 W2 的上下文嵌入（j_embed）
        #self.W2.assign(tf.tensor_scatter_nd_update(self.W2,indexw2,j_embed))
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_j.numpy():
            #print(k)
            self.W2[k,:].assign(j_embed[k1,:]) #把更新后的 j_embed 写回 W2
            k1+=1
        #print(self.W1)
        #print(self.W2)

        return current_loss
    
        
    def run(self,walklen):
        """
        调用随机游走生成训练路径（walks）
        生成正负样本对
        执行多轮迭代训练：
        前向传播 → 计算损失 → 梯度下降更新参数
        每轮记录 loss
        每隔一轮保存当前嵌入
        :param walklen:每次随机游走的最大步数
        :return:
        """
        #self.load_graph(dataset)        
        walks=self.walks(walklen,10) #调用 walks() 方法,walklen 是每次 walk 的长度,10 是每个节点启动 10 次随机游走
        #print(walks)
        pairs,negsamples=self.get_train_pair(walks,2,self.qnum+self.enum)#用 get_train_pair() 从 walk 生成正样本对（context）和负样本列表
        lennegs=len(negsamples)#个重复的节点索引数组，用于负采样
        #print(pairs)

        epochs = range(50)
        loss_total=0
        train_len=len(pairs)
        logfile=open(self.dataset+"/team2box/ExpertsEmbedinglog.txt","w")
        
        for epoch in epochs:  
            #walks=self.walks(walklen)            
            loss=0
            #for k in range(train_len):
            for k in tqdm(range(train_len), desc=f"Epoch {epoch}", ncols=100):
                    #当前正样本对：中心节点 pairs[k][0] 与其上下文 pairs[k][1]
                    tpairs_i=[]
                    tpairs_j=[]
                    tpairs_i.append(pairs[k][0])
                    tpairs_j.append(pairs[k][1])
                    #add negetive samples
                    #negsample=[]
                    nk=0
                    #添加 10 个负样本，与当前中心节点没有边相连 保证负样本不是正样本（正样本已经在 tpairs_j[0]）
                    while nk<10:                        
                        neg=random.randint(0,lennegs-1)                        
                        if negsamples[neg] != tpairs_i and negsamples[neg] not in tpairs_j and negsamples[neg] not in self.G[pairs[k][0]]['n']:
                            tpairs_j.append(negsamples[neg])
                            nk+=1
                    #print(tpairs_i)
                    #print(tpairs_j)
                    #构造输入张量，分别表示中心节点和配对节点（一个正样本 + 若干负样本）
                    inputs_i=tf.Variable(tpairs_i,dtype=tf.int32)
                    inputs_j=tf.Variable(tpairs_j,dtype=tf.int32)
                    #构造标签张量
                    out=np.zeros(shape=(inputs_j.shape[0],1))
                    out[0]=1
                    outputs=tf.Variable(out,dtype=tf.float32)                     
                    loss +=self.train( inputs_i, inputs_j, outputs, learning_rate=0.1)
                    # === 每 500 步记录一次中间日志 ===
                    if k % 500 == 0:
                        log_str = f"[Epoch {epoch}] Step {k}/{train_len} - Partial Loss: {loss.numpy() / (k + 1):.6f}\n"
                        logfile.write(log_str)
                        logfile.flush()
               
            loss_total+=(loss/train_len)
            if epoch%1==0:
                #print('Epoch %2d: loss=%2.5f' % (epoch,  loss_total/(epoch+1)) )  
                l=loss_total/(epoch+1)
                logfile.write("Epoch: "+str(epoch)+" loss: "+str(l.numpy() )+"\n") 
                logfile.flush()    
            if epoch%1==0:               
                self.save_embeddings()
        logfile.close()
    def save_embeddings(self):
        #qfile=open(self.dataset+"/krnmdata1/teamsembeding.txt","w")
        w1=self.W1.numpy()
        w2=self.W2.numpy()
        np.savetxt(self.dataset+"/team2box/expert_question_w1_embedding.txt",w1, fmt='%f')
        np.savetxt(self.dataset+"/team2box/expert_question_w2_embedding.txt",w2, fmt='%f')

dataset=["android","dba","physics","history","mathoverflow"]
data="/home/dyx2/team2box/team2box/data/"+dataset[0]
ob2=ExpertsEmbeding(32,data)
ob2.run(9)
