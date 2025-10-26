from matplotlib.patches import Rectangle
import networkx as nx
import tensorflow as tf
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
import numpy as np
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
#该模型是基于随机游走生成的数据进行训练的，其中目标是学习嵌入，以便在团队网络中表示节点
class Team2Box:
    def __init__(self,hsize,data):
        self.dataset=data 
        #self.CreateTeamG()
        #sys.exit()
        self.G={}  #contains the team network  
        self.Teams=[] 
        self.loadTeams(data)
        self.loadG(data)                
        self.hidden_size=hsize         
        self.W1,self.Offsets=self.weight_variable_teams((self.N,self.hidden_size)) # initilize the first layer weights and offsets
        #print(self.W1,self.Offsets)     
        self.W2=Team2Box.weight_variable((self.N,self.hidden_size))  # initilize the second layer weights      
        #self.displayG()
        #self.displayTeamEmbedding()
    
    def weight_variable(shape):
        """initialize the embedding weights for questions, experts,..."""
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial) 
    
    def weight_variable_teams(self,shape):   
        """initialize the embedding weights of teams"""
        tmp = 2.*np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial=[]#变量初始值
        offsets=[]#变量偏移值
        initial.append(np.random.uniform(-tmp, tmp,shape[1]))
        offsets.append(np.full(shape[1],len(self.Teams[0])/20,dtype=np.float))        
        i=1
        while i<self.N:
            initial.append(np.random.uniform(-tmp, tmp,shape[1]))
            offsets.append(np.full(shape[1],len(self.Teams[i])/20,dtype=np.float))
            i+=1
        return tf.Variable(initial,dtype=tf.float32),tf.Variable(offsets,dtype=tf.float32)
      #tf.Variable TensorFlow 中用于创建可训练的变量的函数。这些变量可以在训练过程中被优化和更新

    def get_train_pair(self,walks,windowsize, N): 
        """Prepar training data"""\
        #用于基于图或词的嵌入模型
        #print(N)
        z=np.zeros((N))
        total=0
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        z1=z/total#将“z”中的每个元素除以“total”，有效计算每个节点出现的概率
        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #probability of keeping a node in the training data
        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)#归一化转换节点计数
        #print(p2)
        negsamples=[]#生成负样本
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
            for context_ind in range(len(walk)):            
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        if (100-p[int(walk[context_ind])]*100)>x:
                            continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])])
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
        """load existing teams"""
        #用于从存储在名为”teams.txt”的文本文件中的数据集中加载现有的团队
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
        """load the team network"""
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
        """get random walks on the team network  用于在以加权图表示的团队网络上生成随机漫步。它使用NetworkX库来读取加权图，使用StellarGraph库来执行有偏随机漫步"""
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/teamsG.txt")        
        rw = BiasedRandomWalk(StellarGraph(G))
        weighted_walks = rw.run(
        nodes=G.nodes(), # root nodes
        length=walklen,    # maximum length of a random walk
        n=5,          # number of random walks per root node 
        p=0.1,         # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,         # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True, #for weighted random walks
        seed=42        # random seed fixed for reproducibility
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
        """Generate team network graph from CQA network"""
        gfile=open(self.dataset+"/CQAG.txt")
        #gfile.readline()
        e=gfile.readline()
        G={}
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
        pfile=open(self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        qnum=int(properties[1])
        anum=int(properties[2])
        enum=int(properties[3])
        T=[]
        Tq=[]
        EBQ={} # for each expert save question id with best answer and its score 
        EBQteam=[]
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
        tfile=open(self.dataset+"/teams.txt","w")
        tfile.write("teamID expertID expertID ...\n")
        tqfile=open(self.dataset+"/teamsquestions.txt","w")
        tqfile.write("teamID questionID questionID ...\n")
        qfile=open(self.dataset+"/teamsG.txt","w")
        lenT=len(T)
        
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
        for e in EBQ:
            ebfile.write(str(e)+" "+ str(indx.index(qT[EBQ[e]['q']]))+" "+str(EBQ[e]['q'])+" "+str(EBQ[e]['s'])+"\n") 
        ebfile.close()
        tfile.close()  
        tqfile.close()
        qfile.close()
        print("done!!!!!!1")
        
    def displayTeamEmbedding(self):  
        """Visualize the embeddings of teams"""
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
        plt.show();
    
    def loss(predicted_y, target_y):
        #print("loss=",predicted_y,target_y)        
        
        loss=tf.square(predicted_y[0,0]-target_y[0,0])+tf.reduce_mean(tf.square(tf.nn.relu(target_y[1:,0]-predicted_y[1:,0])))
        #print(loss)
        return loss
        #return tf.reduce_mean(tf.square(predicted_y - target_y))
    
    def loss_min(self):
        #print("loss=",predicted_y,target_y)        
        self.predicted_y=self.model(self.inputs_i, self.inputs_j)
        self.curr_loss=tf.square(self.predicted_y[0,0]-self.target_y[0,0])+tf.reduce_mean(tf.square(tf.nn.relu(self.target_y[1:,0]-self.predicted_y[1:,0])))
        #print(loss)
        return self.curr_loss    

    def model(self,inputs_i,inputs_j):    
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
        for k in inputs_i.numpy():
            #print(k)
            self.W1[k,:].assign(i_embed[k1,:])
            k1+=1        
        
        k1=0
        #print(inputs_j.numpy())
        j_embed=j_embed-(learning_rate * dW1.values[1:,:])
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
            loss=0
            for k in range(train_len):
                    tpairs_i=[]
                    tpairs_j=[]
                    tpairs_i.append(pairs[k][0])
                    tpairs_j.append(pairs[k][1])
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
                    i_offset=tf.nn.embedding_lookup(self.Offsets,inputs_i).numpy()
                    j_offset=tf.nn.embedding_lookup(self.Offsets,inputs_j).numpy()
                    offsets=j_offset[:,0]+i_offset[0,0]                  
                    indj=self.G[inputs_i.numpy()[0]]['n'].index(inputs_j.numpy()[0])
                    offsets[0]=(1-self.G[inputs_i.numpy()[0]]['w'][indj])*offsets[0]
                    
                    outputs=tf.Variable(offsets,dtype=tf.float32) 
                    outputs=tf.reshape(outputs,(outputs.shape[0],1))
                   # print("out=",outputs)
                    #print("current_loss= %2.5f"%(current_loss))
                    loss+=self.train( inputs_i, inputs_j, outputs, learning_rate=0.1)
                #if i%100==0:
                #    print('Epoch %2d: Node %4d: loss=%2.5f' % ( epoch, i ,  loss))
                    #print(self.W1)
            loss_total+=(loss/train_len)
            l=loss_total/(epoch+1)
            logfile.write("Epoch: "+str(epoch)+" loss: "+str(l.numpy() )+"\n")
            logfile.flush()
            if epoch%5==0:
               self.save_team_embedding()
            
        logfile.close()
    def save_team_embedding(self):
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
data="/home/pjm/projects/team2box/data/"+dataset[0]
ob=Team2Box(32,data)
#ob=Team2Box(32,dataset[3])
ob.run(5)  

