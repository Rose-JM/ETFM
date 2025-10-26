
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from matplotlib.patches import Rectangle
import networkx as nx
from tqdm import tqdm
from datetime import datetime


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
import numpy as np
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
"""
构造团队图并学习每个团队的中心位置与半径，学习团队嵌入表示，在低维空间中反映团队间的结构关系。
| 模块      |                    说明                              |
| ---------| ----------------------------------------------------- |
| 团队图构建 | 将回答过相同问题的专家划为一个团队，并建立团队之间的“相交图”     |
| 向量初始化| 使用正态分布随机初始化团队嵌入中心和偏移半径                   |
| 随机游走   | 使用 `BiasedRandomWalk` 模拟局部图结构                   |
| 嵌入训练   | 结合正负样本进行距离学习（越近表示越相关）                    |
| 嵌入保存   | 将训练好的嵌入写入文件供下游使用                            |
"""
class Team2Box:
    def __init__(self,hsize,data):
        """
        初始化构造函数
        :param hsize:嵌入维度
        :param data:数据集路径
        """
        self.dataset=data 
        #self.CreateTeamG()
        #sys.exit()
        self.G={}        #图结构，存储团队间连接信息
        self.Teams=[]       # 所有团队的专家成员列表
        self.loadTeams(data)    # 加载团队信息
        self.loadG(data)        # 加载团队网络图
        self.hidden_size=hsize         # 嵌入维度

        # 初始化嵌入中心（W1）和偏移（Offsets）
        self.W1,self.Offsets=self.weight_variable_teams((self.N,self.hidden_size))
        #print(self.W1,self.Offsets)
        # 初始化另一组嵌入（W2），用于可选的双向训练结构
        self.W2=Team2Box.weight_variable((self.N,self.hidden_size))
        #self.displayG()
        #self.displayTeamEmbedding()
    
    def weight_variable(shape):
        """初始化 questions, experts的权重"""
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial) 
    
    def weight_variable_teams(self,shape):   
        """初始化 team 团队的嵌入中心和偏移值"""
        tmp = 2.*np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial=[]      # 嵌入中心向量
        offsets=[]      #嵌入偏移向量
        initial.append(np.random.uniform(-tmp, tmp,shape[1]))#嵌入中心随机初始化
        offsets.append(np.full(shape[1],len(self.Teams[0])/20,dtype=np.float)) #偏移向量，每个维度都用相同的值填充，该值为“团队规模除以 20”
        i=1
        while i<self.N:
            initial.append(np.random.uniform(-tmp, tmp,shape[1]))
            offsets.append(np.full(shape[1],len(self.Teams[i])/20,dtype=np.float))
            i+=1
        return tf.Variable(initial,dtype=tf.float32),tf.Variable(offsets,dtype=tf.float32) # tf.Variable TensorFlow 中用于创建可训练的变量的函数。这些变量可以在训练过程中被优化和更新

    def get_train_pair(self,walks,windowsize, N):
        """
        构造训练正负样本对
        1.正样本对：
        对每个游走路径 walk = [t1, t2, t3, ..., tn]
        对每个位置 context_ind（如 t2）
        在窗口 window_size=1 范围内，取前一个、后一个节点当正样本：正样本对示例：(t2, t1), (t2, t3)
        要考虑 “节点保留概率 p”，有时跳过某些样本（用于降噪）

        2.负样本对：基于频率采样分布构造
        每个节点 t 出现次数越多，越有可能被采样
        使用 0.75 次幂平滑节点频率
        构造一个“采样池” negsamples，从中随机抽样，组成负样本对 (t2, tn)，其中 tn 和 t2 没有关系
        ：return:
            pairs: 所有正样本对；
            negs: 负样本池（后续训练时每个正样本随机配若干负样本）
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
        z1=z/total      #将“z”中的每个元素除以“total”，有效计算每个节点出现的概率
        z1[z1 == 0] = 1e-8  # 避免除0
        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #probability of keeping a node in the training data
        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)        #归一化转换节点计数
        #print(p2)
        negsamples=[]
        #生成负样本
        maxrep=0
        for i in range(N):
            rep=int(p2[i]*100)             
            if rep==0:
                rep=1
            if maxrep <rep:
                maxrep=rep
            if i not in self.G:
                rep=maxrep*10
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        for i in range(len(walks)):
            walk=walks[i]
            #对每一个目标节点 walk[context_ind]，定义一个上下文窗口 [start, context_ind) 与 [context_ind+1, context_ind+windowsize]
            for context_ind in range(len(walk)):            
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):
                    #从左窗口生成正样本对
                    #如果节点未被随机丢弃（根据概率 p），就生成一个正样本；正样本是：中心节点 vs 窗口内上下文节点
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        if (100-p[int(walk[context_ind])]*100)>x:
                            continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])])
                        #从右窗口生成正样本对
                        if i+context_ind+1<len(walk):
                            x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                            if (100-p[int(walk[context_ind])]*100)>x:
                                continue
                            if  walk[context_ind]!=walk[i+context_ind+1]:   
                                pairs.append([int(walk[context_ind]),int(walk[i+context_ind+1])])
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    
    def loadTeams(self,dataset):
        """从文件中加载团队结构"""
        self.dataset=dataset
        gfile=open(dataset+"/teams.txt")
        gfile.readline()
        e=gfile.readline()
        self.Teams=[]
        while e:
            ids=e.strip().split(" ")
            i=int(ids[0])                    
            if i not in self.Teams:
                self.Teams.append([])            
            for j in ids[1:]:    
                        self.Teams[i].append(int(j))                  
            e=gfile.readline()
        #print(self.Teams)
        self.N=len(self.Teams)
        print("N=",self.N)
        gfile.close() 
        
    def loadG(self,dataset):
        """加载团队网络"""
        self.dataset=dataset
        gfile=open(dataset+"/teamsG.txt")
        #gfile.readline()
        e=gfile.readline()
        self.G={}
        ecount=1
        while e:
            ids=e.strip().split(" ")
            #print(ids)
            i=int(ids[0])
            j=int(ids[1])
            w=float(ids[2])            
            if i not in self.G:
                self.G[i]={'n':[],'w':[]}            
            if j not in self.G:    
                        self.G[j]={'n':[],'w':[]}                    
            self.G[i]['n'].append(j)
            self.G[i]['w'].append(w)            
            self.G[j]['n'].append(i)
            self.G[j]['w'].append(w)
            e=gfile.readline()
            ecount+=1
        lenG=len(self.G)
        print("#teams with no intersections: ",self.N-lenG)
        print("#edges",ecount)
        #print(self.G)
        gfile.close() 
    
    def walks(self,walklen):
        """
        在以加权折叠图表示的团队网络上生成随机漫步。
        使用NetworkX库来读取加权图
        使用StellarGraph库来执行有偏随机漫步
        """
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/teamsG.txt")        
        rw = BiasedRandomWalk(StellarGraph(G))
        weighted_walks = rw.run(
        nodes=G.nodes(), # 所有团队作为起点
        length=walklen,    # 每条游走的长度
        n=5,           # 每个节点执行几次游走
        p=0.1,         # 返回概率，越小越容易回头（即 BFS 风格）
        q=2.0,         # 远离概率，越大越倾向跳出（即 DFS 风格）
        weighted=True, #使用边的权重作为跳转概率依据
        seed=42        #随机种子
        )
        print("Number of random walks: {}".format(len(weighted_walks)))
        print(weighted_walks[0:10])               
        return weighted_walks      
    
    def displayG(self):
        """Used to visualize the team network"""
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/teamsG.txt")
        nodes=list(G.nodes())
        #print(nodes)
        for i in range(self.N):
            if str(i) not in nodes:
                G.add_node(i)
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
        pos = nx.spring_layout(G)  # positions for all nodes
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=300)
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)
        # labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        plt.axis('off')
        plt.show()
    
    def CreateTeamG(self): 
        """
        从HCN中提取出专家集合，将其视为一个“团队”。
        合并成员完全一致的团队。
        计算 Jaccard 相似度，如果有重叠就建立边，边权代表重合度，从而构建出一个“团队折叠图”。
        用于嵌入模型训练，模拟团队之间的结构关系
        输出：
        - teams.txt         （团队 → 专家）
        - teamsquestions.txt（团队 → 问题）
        - teamsG.txt        （团队图）
        - ExpertBestQuetionAnswer.txt（专家在何处表现最佳）
        """
        gfile=open(self.dataset+"/CQAG.txt")
        #gfile.readline()
        e=gfile.readline()
        G={}
        #构建整个专家图结构，存在 G 中
        while e:
            ids=e.strip().split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=float(ids[2])            
            if i not in G:
                G[i]={'n':[],'w':[]}            
            if j not in G:    
                        G[j]={'n':[],'w':[]}                    
            G[i]['n'].append(j)
            G[i]['w'].append(w)            
            G[j]['n'].append(i)
            G[j]['w'].append(w)
            e=gfile.readline()
        N=len(G)
        print(N)        
        gfile.close()       
        #print(G)
        #读取图的节点数量信息（问题/回答/专家数量）
        pfile=open(self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        N=int(properties[0]) #  N=|Qestions|+|Answers|+|Experts|
        qnum=int(properties[1])
        anum=int(properties[2])
        enum=int(properties[3])
        T=[]    #问题 i 下所有参与回答的专家
        Tq=[]   #表示第 i 个团队包含的问题编号
        EBQ={}  #记录每位专家在哪个问题下给出了最佳答案
        EBQteam=[]

        #遍历所有问题节点，构建团队专家集合
        #对每个问题 i：查找其回答者（间接找专家）；将这些专家组成团队 T[i]；同时更新 EBQ：某个专家在哪个问题上的回答评分最高。
        for i in range(qnum):
            T.append([])
            Tq.append([i])
            for k in range(len(G[i]['n'])):
                a=G[i]['n'][k]
                s=G[i]['w'][k]
                for e in G[a]['n']:
                    if i!=e:
                        T[i].append(e)
                    if i!=e and e not in EBQ:
                        EBQ[e]={'q':i,'s':s}
                    elif i!=e:
                        if s>EBQ[e]['s']:
                            EBQ[e]['q']=i
                            EBQ[e]['s']=s 
                        
        #print(EBQ)
        lenT=len(T)
        qT=list(range(lenT))
        #print("qT",qT)
        flag=np.zeros(lenT)
        #合并专家集合完全相同的问题（共享专家 → 同一团队）
        for i in range(lenT):            
            j=i+1
            while(j<lenT):
                if flag[j]==0 and (set(T[i])==set(T[j])):
                    flag[j]=1
                    Tq[i].append(j)
                    Tq[j].append(i) 
                    qT[j]=i
                    
                j+=1                    
        #print (flag)
        #print(Tq)
        #print("qT=",qT)
        T=np.array(T)
        T,indx=np.unique(T,return_index=True)
        print(T,indx)
        indx=list(indx)
        #输出团队成员 & 所属问题文件
        tfile=open(self.dataset+"/teams.txt","w") #每一行为一个团队，其成员是多个专家 ID
        tfile.write("teamID expertID expertID ...\n")
        tqfile=open(self.dataset+"/teamsquestions.txt","w")#每一行为一个团队，列出其包含的问题 ID
        tqfile.write("teamID questionID questionID ...\n")
        qfile=open(self.dataset+"/teamsG.txt","w")
        lenT=len(T)
        #构建团队折叠图图（teamsG.txt）
        #若两个团队的成员有重叠，就连一条边；边权重使用 Jaccard 相似度估计；得到团队图 teamsG.txt
        for i in range(lenT):
            i_index=indx[i]
            tfile.write(str(i))
            tqfile.write(str(i))           
            for e in T[i]:
                tfile.write(" "+str(e))
            
            for q in Tq[i_index]:
                tqfile.write(" "+str(q))
            tfile.write("\n") 
            tqfile.write("\n")
            
            if i+1!=lenT:
                #qfile.write(str(i))
                j=i+1
                while(j<lenT):  
                    #if j in indx:
                    #j_index=indx[j]
                    c=set(T[i]).intersection(set(T[j]))
                    if len(c)>0:                  
                        wi=1.0*(len(c)/(len(T[i])+len(T[j])-len(c)))  # comute the weights in the team network 
                        qfile.write(str(i)+" "+str(j)+" "+str(wi)+"\n")
                    j+=1             
                #qfile.write("\n")      
        ebfile=open(self.dataset+"/ExpertBestQuetionAnswer.txt","w")
        ebfile.write("ExpertID   TeamID    QuestionIDwithBestAnswer    Score \n")
        #输出专家-最佳团队-问题文件
        for e in EBQ:
            ebfile.write(str(e)+" "+ str(indx.index(qT[EBQ[e]['q']]))+" "+str(EBQ[e]['q'])+" "+str(EBQ[e]['s'])+"\n") 
        ebfile.close()
        tfile.close()  
        tqfile.close()
        qfile.close()
        print("done!!!!!!1")
        
    def displayTeamEmbedding(self):  
        """可视化团队嵌入"""
        Centers=self.W1.numpy().copy()
        Offsets=self.Offsets.numpy().copy()
        #print(embed)
        #model = TSNE(n_components=2, random_state=0)
        #y=model.fit_transform(embed) 
        y=Centers
        max1=np.amax(y,axis=0)+1.1*np.amax(Offsets,axis=0)
        min1=np.amin(y,axis=0)-1.1*np.amax(Offsets,axis=0)
        
        plt.figure(figsize=(5,5))
        plt.plot(y[0:self.N,0],y[0:self.N,1],'r+');
        
        for i in range(self.N):
            plt.text(y[i,0],y[i,1], i, fontsize=8)  
        
        ax = plt.gca()    
        ax.set_aspect(1)
        for i in range(self.N):
            #c= Rectangle((y[i,0]-Offsets[i,0],y[i,1]-Offsets[i,0]), 2*Offsets[i,0],2*Offsets[i,0] , linewidth=1,edgecolor='r',facecolor='none')
            c=plt.Circle((y[i,0], y[i,1]), Offsets[i,0], color='b', clip_on=False,fill=False)
            ax.add_patch(c)
        
        ax.set_xlim([min1[0], max1[0]])
        ax.set_ylim([min1[1], max1[1]])
        plt.show()
    
    def loss(predicted_y, target_y):
        """损失函数：正样本距离应接近，负样本远离"""
        #print("loss=",predicted_y,target_y)        
        
        loss=tf.square(predicted_y[0,0]-target_y[0,0])+tf.reduce_mean(tf.square(tf.nn.relu(target_y[1:,0]-predicted_y[1:,0])))
        #print(loss)
        return loss
        #return tf.reduce_mean(tf.square(predicted_y - target_y))
    
    def loss_min(self):
        """
        将 loss() 封装成可以自动求导的形式
        :return:
        """
        #print("loss=",predicted_y,target_y)        
        self.predicted_y=self.model(self.inputs_i, self.inputs_j)
        self.curr_loss=tf.square(self.predicted_y[0,0]-self.target_y[0,0])+tf.reduce_mean(tf.square(tf.nn.relu(self.target_y[1:,0]-self.predicted_y[1:,0])))
        #print(loss)
        return self.curr_loss    

    def model(self,inputs_i,inputs_j):
        """计算节点对之间的欧几里得距离"""
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W1, inputs_j, name='jemb')          
        # Learning-To-Rank layer. o is the final matching score.
        
        #print(i_offset[0,0],j_offset[0,0])
        #print("offs")
        #print(i_offset,j_offset)      
        #print("offsets=",offsets)
        c=tf.square(tf.subtract(i_embed,j_embed))         
        d=tf.sqrt(tf.reduce_sum(c,axis=1))
        #print("d=",d)
               
        #d=(offsets-d)/offsets
        #print(inputs_i,inputs_j)
        #print("d=",d)
        o=tf.reshape(d,(d.shape[0],1))       
        #print(o)
        return o
    
    def train(self, inputs_i, inputs_j, outputs, learning_rate):
        """
        更新参数的实现
        :param inputs_i:正样本中心
        :param inputs_j:上下文 + 负样本
        :param outputs:期望距离
        :param learning_rate:学习率
        :return:
        """
        #print("inputs=",inputs_i,inputs_j)
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W1, inputs_j, name='jemb')  
        with tf.GradientTape() as t:
            current_loss = Team2Box.loss(self.model(inputs_i,inputs_j), outputs)
        dW1, dW2 = t.gradient(current_loss, [self.W1, self.W2]) 
        #print("dw1=",dW1)
        #print("dw2=",dW2)
        i_embed=i_embed-(learning_rate * dW1.values[0,:])
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_i.numpy():#更新中心节点 i 的向量
            #print(k)
            self.W1[k,:].assign(i_embed[k1,:])
            k1+=1        
        
        k1=0
        #print(inputs_j.numpy())
        j_embed=j_embed-(learning_rate * dW1.values[1:,:])#更新正负样本节点 j 的嵌入
        for k in inputs_j.numpy():
            #print(k)
            self.W1[k,:].assign(j_embed[k1,:])
            k1+=1
        return current_loss
    
    def run_adam(self,walklen):
        """train the model using ADAM optimizer"""
        #self.load_graph(dataset)        
        walks=self.walks(walklen)
        pairs,negsamples=self.get_train_pair(walks,1,self.N)
        lennegs=len(negsamples)
        
        epochs = range(51)
        loss_total=0
        train_len=len(pairs)
        #opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #opt=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
        opt=tf.keras.optimizers.Nadam(learning_rate=0.02, beta_1=0.9, beta_2=0.999)
        #opt=tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        for epoch in epochs:  
            #walks=self.walks(walklen)            
            loss=0
            for k in range(train_len):
                    tpairs_i=[]
                    tpairs_j=[]
                    tpairs_i.append(pairs[k][0])
                    tpairs_j.append(pairs[k][1])
                    #add negetive samples
                    #negsample=[]
                    nk=0
                    while nk<1:                        
                        neg=random.randint(0,lennegs-1)                        
                        if negsamples[neg] != tpairs_i and negsamples[neg] not in tpairs_j and negsamples[neg] not in self.G[pairs[k][0]]['n']:
                            tpairs_j.append(negsamples[neg])
                            nk+=1
                    #print(tpairs_i)
                    #print(tpairs_j)
                    self.inputs_i=tf.Variable(tpairs_i,dtype=tf.int32)
                    self.inputs_j=tf.Variable(tpairs_j,dtype=tf.int32)
                    i_offset=tf.nn.embedding_lookup(self.Offsets,self.inputs_i).numpy()
                    j_offset=tf.nn.embedding_lookup(self.Offsets,self.inputs_j).numpy()
                    offsets=j_offset[:,0]+i_offset[0,0]                  
                    indj=self.G[self.inputs_i.numpy()[0]]['n'].index(self.inputs_j.numpy()[0])
                    offsets[0]=(1-self.G[self.inputs_i.numpy()[0]]['w'][indj])*offsets[0]
                    
                    outputs=tf.Variable(offsets,dtype=tf.float32) 
                    self.target_y=tf.reshape(outputs,(outputs.shape[0],1))
                    
                    
                    opt.minimize(self.loss_min, var_list=[self.W1])
                
                    # print("out=",outputs)
                    #print("current_loss= %2.5f"%(current_loss))
                    loss+=self.curr_loss
                #if i%100==0:
                #    print('Epoch %2d: Node %4d: loss=%2.5f' % ( epoch, i ,  loss))
                    #print(self.W1)
            loss_total+=(loss/train_len)
            print('Epoch %2d: loss=%2.5f' % (epoch,  loss_total/(epoch+1)) )      
            if epoch%20==0:                
                self.displayG()
                self.displayTeamEmbedding()
    
    def run(self,walklen):
        """训练入口"""
        #self.load_graph(dataset)        
        walks=self.walks(walklen)
        #print(walks)
        pairs,negsamples=self.get_train_pair(walks,1,self.N)
        #print(negsamples)
        lennegs=len(negsamples)
        
        epochs = range(101)
        loss_total=0
        train_len=len(pairs)
        logfile=open(self.dataset+"/team2box/Team2boxlog.txt","w")
        
        for epoch in epochs:  
            #walks=self.walks(walklen)
            # === 1. 小批量采样训练对 ===
            sample_size = 5000
            sample_idx = np.random.choice(train_len, size=min(sample_size, train_len), replace=False)
            sampled_pairs = pairs[sample_idx]

            loss=0
            #for k in range(train_len):
            for k in tqdm(range(train_len), desc=f"Epoch {epoch}", ncols=100):
                    tpairs_i=[]
                    tpairs_j=[]
                    tpairs_i.append(pairs[k][0])
                    tpairs_j.append(pairs[k][1])
            # for k in tqdm(range(len(sampled_pairs)), desc=f"Epoch {epoch}", ncols=100):
            #         tpairs_i=[]
            #         tpairs_j=[]
            #         tpairs_i.append(sampled_pairs[k][0])
            #         tpairs_j.append(sampled_pairs[k][1])
                    #add negetive samples
                    #negsample=[]
                    nk=0
                    while nk<10:                        
                        neg=random.randint(0,lennegs-1)                        
                        if negsamples[neg] != tpairs_i and negsamples[neg] not in tpairs_j and negsamples[neg] not in self.G[pairs[k][0]]['n']:
                            tpairs_j.append(negsamples[neg])
                            nk+=1
                    #print(tpairs_i)
                    #print(tpairs_j)
                    inputs_i=tf.Variable(tpairs_i,dtype=tf.int32)
                    inputs_j=tf.Variable(tpairs_j,dtype=tf.int32)
                    #如果 i 和 j 在图中有边，则希望它们之间的距离小于半径之和
                    i_offset=tf.nn.embedding_lookup(self.Offsets,inputs_i).numpy()
                    j_offset=tf.nn.embedding_lookup(self.Offsets,inputs_j).numpy()
                    offsets=j_offset[:,0]+i_offset[0,0]                  
                    indj=self.G[inputs_i.numpy()[0]]['n'].index(inputs_j.numpy()[0])
                    #样本的距离目标根据边权重调整，权重越大（越强），希望距离越小
                    offsets[0]=(1-self.G[inputs_i.numpy()[0]]['w'][indj])*offsets[0]
                    
                    outputs=tf.Variable(offsets,dtype=tf.float32) 
                    outputs=tf.reshape(outputs,(outputs.shape[0],1))
                   # print("out=",outputs)
                    #print("current_loss= %2.5f"%(current_loss))
                    loss+=self.train( inputs_i, inputs_j, outputs, learning_rate=0.1)
                    # === 每 500 步记录一次中间日志 ===
                    if k % 500 == 0:
                        log_str = f"[Epoch {epoch}] Step {k}/{train_len} - Partial Loss: {loss.numpy() / (k + 1):.6f}\n"
                        logfile.write(log_str)
                        logfile.flush()
                #if i%100==0:
                #    print('Epoch %2d: Node %4d: loss=%2.5f' % ( epoch, i ,  loss))
                    #print(self.W1)
            loss_total+=(loss/train_len)
            l=loss_total/(epoch+1)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] Epoch {epoch} | Loss: {l.numpy():.6f} | Samples: {train_len}\n"
            logfile.write(log_line)
            logfile.flush()
            if epoch%5==0:
               self.save_team_embedding()
        logfile.close()

    def save_team_embedding(self):
        """
        保存嵌入中心向量和偏移向量
        :return:
        """
        #qfile=open(self.dataset+"/krnmdata1/teamsembeding.txt","w")
        w1=self.W1.numpy()
        offsets=self.Offsets.numpy()
        #np.savetxt(self.dataset+"/team2box/teamsembeding.txt",w1, fmt='%f')
        #np.savetxt(self.dataset+"/team2box/teamsOffsets.txt",offsets, fmt='%f')
        np.savetxt(self.dataset+"/team2box/teamsembeding_0.txt",w1, fmt='%f')
        np.savetxt(self.dataset+"/team2box/teamsOffsets_0.txt",offsets, fmt='%f')
        #qfile.close()
        
      
dataset=["android","dba","physics","mathoverflow","history"]
#dataset=["android"]
#data="D:/pycharm_project/team2box-master/team2box-master/data/"+dataset[3]
#data="D:/pycharm_project/team2box-master/team2box-master/data/"+dataset[0]
data="/home/dyx2/team2box/team2box/data/"+dataset[0]
ob=Team2Box(32,data)
#ob=Team2Box(32,dataset[3])
ob.run(5)  

