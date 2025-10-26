import numpy as np
import random
import sys
import math
import datetime as datetime
class TeamFormation:        
    def __init__(self,path,data):  
        self.path=path
        self.dataset=data        
        pfile=open(self.path+self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        self.qnum=int(properties[1]) #number of questions
        self.anum=int(properties[2]) #number of answers
        self.enum=int(properties[3]) #number of answerers
        
        self.Expert_id_map={} #used to map answerers ids to original ones
        pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
        pufile.readline()
        line=pufile.readline().strip()
        eids=[]
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        for i in range(len(eids)):
            self.Expert_id_map[i]=eids[i]
        
    def compare_EF(self,teamsembeding,teamsOffsets,team2box_w1_embedding,n2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteams,rest2b,resm2v,resne,ressq): 
        """对比多种专家团队推荐方法的效果，并保存评估指标结果"""
        self.GoldStandard_teams=self.LoadGoldStandard()    #load gold standard teams    
        #print(self.GoldStandard_teams)
        #sys.exit(0)
        self.testquestions=self.loadtestresults()
        #print(self.testquestions)  
        self.usertags=self.loadusertags() 
        #print(self.usertags)
        self.questiontags=self.loadquestiontags() 
        #print(self.questiontags)
        #sys.exit(0)
        
        #lists to save results
        All_SC_t2b, All_EL_t2b, All_CL_t2b, All_GM_t2b=[], [], [], []        
        All_SC_m2v, All_EL_m2v, All_CL_m2v, All_GM_m2v=[], [], [], []     
        All_SC_ne, All_EL_ne, All_CL_ne, All_GM_ne=[], [], [], []
        All_SC_sq, All_EL_sq, All_CL_sq, All_GM_sq=[], [], [], []
        maxteamsize=6   # forms teams with sizes 1,2,...,6  
          
        for teamsize in range(1, maxteamsize+1):
                #load teams, questions, experts, embeddings obtained by t2b
                self.teamcenters=np.loadtxt(self.path+self.dataset+teamsembeding)
                #print(self.teamcenters)
                self.teamoffsets=np.loadtxt(self.path+self.dataset+teamsOffsets)
                #print(self.teamoffsets)
                self.eqembedding=np.loadtxt(self.path+self.dataset+team2box_w1_embedding)
                self.numq2map=Numq2map  #set between 1 .. n 
                #print("num q to map: "+str(self.numq2map)) 

                #compute metrics on t2b results    
                fout=open("detailedresults/team2box_results_"+self.dataset+"_"+str(teamsize)+".txt","w") 
                self.testqmap2embeddings=self.maptestqtoembeddingspace(self.eqembedding,fout)     
                self.testqtopteams=self.findtopkteams(Numtopkteams,fout)
                self.loadTeams()
                self.testq_teamsize=self.getteamsize()              
                print("team size:"+str(teamsize)) 
                #print("team2box results:")    
                self.testq_teams,lenes=self.formteamt2b(teamsize,fout)#form teams
                SC,EL,CL,GM=self.displayteams(self.testq_teams,fout) #compute metrics
                All_SC_t2b.append(SC)
                All_EL_t2b.append(EL)
                All_CL_t2b.append(CL)
                All_GM_t2b.append(GM)
                strtable3="\\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}"
                outputfile.write("\nTeam2box:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}")  
                print("\nTeam2box:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}")  

                #compute metrics on m2v results
                fout=open("detailedresults/meatpath_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
                self.n2veqembedding=np.loadtxt(self.path+self.dataset+n2v_w1_embedding)             
                self.testqmap2n2vembeddings=self.maptestqtoembeddingspace(self.n2veqembedding,fout)
                #print("\nm2v results:")
                self.n2vtestq_teams=self.n2vformteam(teamsize)  #form teams            
                SC,EL,CL,GM=self.displayteams(self.n2vtestq_teams,fout)  #compute metrics
                All_SC_m2v.append(SC)
                All_EL_m2v.append(EL)
                All_CL_m2v.append(CL)
                All_GM_m2v.append(GM)
                outputfile.write("\nmetapath:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
                print("\nmetapath:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
                strtable1=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))           
                #print("+++++++++++++++++++++++++++++++++++++++++++++")            

                #compute metrics on NeRank results     
                self.nerankteams=self.NeRank_formteam(teamsize)#form teams
                fout=open("detailedresults/nerank_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
                SC,EL,CL,GM=self.displayteams(self.nerankteams,fout) #compute metrics
                All_SC_ne.append(SC)
                All_EL_ne.append(EL)
                All_CL_ne.append(CL)
                All_GM_ne.append(GM)
                outputfile.write("\nNeRank:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
                print("\nNeRank:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
                strtable2=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))
                #outputfile.write("\n"+strtable1+"   &        "+strtable2+"    &           "+strtable3)

                #compute metrics on Seq results
                self.seqpointwise_teams=self.seqpointwise_formteam(teamsize) #form teams
                fout=open("detailedresults/seq_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
                SC,EL,CL,GM=self.displayteams(self.seqpointwise_teams,fout) #compute metrics
                All_SC_sq.append(SC)
                All_EL_sq.append(EL)
                All_CL_sq.append(CL)
                All_GM_sq.append(GM)
                outputfile.write("\nseqpointwise:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
                print("\nseqpointwise:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
                strtable4=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))
        
        # display and save results
        print("\nt2b:")
        outputfile.write("\nt2b:")
        print("SC=",All_SC_t2b)
        outputfile.write("\nSC="+str(All_SC_t2b))
        print("EL=",All_EL_t2b)
        outputfile.write("\nEL="+str(All_EL_t2b))
        print("CL=[",All_CL_t2b)
        outputfile.write("\nCL=["+str(All_CL_t2b))
        print("GM=",All_GM_t2b) 
        outputfile.write("\nGM="+str(All_GM_t2b)) 
        
        print("\nm2v:")
        outputfile.write("\nm2v:")
        print("SC=",All_SC_m2v)
        outputfile.write("\nSC="+str(All_SC_m2v))
        print("EL=",All_EL_m2v)
        outputfile.write("\nEL="+str(All_EL_m2v))
        print("CL=",All_CL_m2v)
        outputfile.write("\nCL="+str(All_CL_m2v))
        print("GM=",All_GM_m2v) 
        outputfile.write("\nGM="+str(All_GM_m2v) )
        
        print("\nNeRank:")
        outputfile.write("\nNeRank:")
        print("SC=",All_SC_ne)
        outputfile.write("\nSC="+str(All_SC_ne))
        print("EL=",All_EL_ne)
        outputfile.write("\nEL="+str(All_EL_ne))
        print("CL=",All_CL_ne)
        outputfile.write("\nCL="+str(All_CL_ne))
        print("GM=",All_GM_ne)
        outputfile.write( "\nGM="+str(All_GM_ne))
        
        print("\nSeq:")
        outputfile.write("\nSeq:")
        print("SC=",All_SC_sq)
        outputfile.write("\nSC="+str(All_SC_sq))
        print("EL=",All_EL_sq)
        outputfile.write("\nEL="+str(All_EL_sq))
        print("CL=",All_CL_sq)
        outputfile.write("\nCL="+str(All_CL_sq))
        print("GM=",All_GM_sq) 
        outputfile.write("\nGM="+str(All_GM_sq)) 
        
        
        outputfile.write("\nt2b:")        
        outputfile.write("\n"+"["+str(All_SC_t2b)+",\n"+str(All_EL_t2b)+",\n"+str(All_CL_t2b)+",\n"+str(All_GM_t2b)+"]")            
        outputfile.write("\nm2v:")
        outputfile.write("\n"+"["+str(All_SC_m2v)+",\n"+str(All_EL_m2v)+",\n"+str(All_CL_m2v)+",\n"+str(All_GM_m2v)+"]")       
        outputfile.write("\nNeRank:")
        outputfile.write("\n"+"["+str(All_SC_ne)+",\n"+str(All_EL_ne)+",\n"+str(All_CL_ne)+",\n"+str(All_GM_ne)+"]")
        outputfile.write("\nSeq:")
        outputfile.write("\n"+"["+str(All_SC_sq)+",\n"+str(All_EL_sq)+",\n"+str(All_CL_sq)+",\n"+str(All_GM_sq)+"]")
        rest2b.append([All_SC_t2b,All_EL_t2b,All_CL_t2b,All_GM_t2b])
        resm2v.append([All_SC_m2v,All_EL_m2v,All_CL_m2v,All_GM_m2v])
        resne.append([All_SC_ne,All_EL_ne,All_CL_ne,All_GM_ne])
        ressq.append([All_SC_sq,All_EL_sq,All_CL_sq,All_GM_sq])
        
    def compare_TF(self,teamsembeding,teamsOffsets,team2box_w1_embedding,n2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteams): 
        """
        对比传统团队构建方法（TF方法）的效果。通过让 T2B 模拟这些传统方法所使用的团队规模，然后在相同条件下进行性能对比
        """
        self.GoldStandard_teams=self.LoadGoldStandard()
        #加载标准团队数据
        #print(self.GoldStandard_teams)
        #sys.exit(0)
        self.testquestions=self.loadtestresults()
        #加载测试结果
        #print(self.testquestions)  
        self.usertags=self.loadusertags()
        #加载用户标签数据
        #print(self.usertags)
        self.questiontags=self.loadquestiontags()
        #加载问题标签
        #print(self.questiontags)       
        
        # load team centers and offsets
        self.teamcenters=np.loadtxt(self.path+self.dataset+teamsembeding)
        #print(self.teamcenters)
        self.teamoffsets=np.loadtxt(self.path+self.dataset+teamsOffsets)
        #print(self.teamoffsets)
        self.eqembedding=np.loadtxt(self.path+self.dataset+team2box_w1_embedding)
        self.numq2map=Numq2map  #set between 1 .. n 
        
        #print("num q to map: "+str(self.numq2map))    
        fout=open("detailedresults/team2boxvsTF_results_"+self.dataset+"_"+str(teamsize)+".txt","w") 
        self.testqmap2embeddings=self.maptestqtoembeddingspace(self.eqembedding,fout)
        #将测试问题映射到嵌入空间
        self.testqtopteams=self.findtopkteams(Numtopkteams,fout)
        self.loadTeams()
        self.testq_teamsize=self.getteamsize()
        
        
        self.ccteams,av_teamsize=self.loadteams("allcc")
        print("\nCC results:")        
        SC,EL,CL,GM=self.displayteams_TF(self.ccteams,fout)
        print(SC,EL,CL,GM) 
        outputfile.write("\nCC SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+" GM="+str(GM)+"\n")       
        print("team size:"+str(av_teamsize)) 
        #print("team2box results:")    
        self.testq_teams,lenes=self.formteamt2b(av_teamsize,fout)
        SC_t2b,EL_t2b,CL_t2b,GM_t2b=self.displayteams(self.testq_teams,fout)        
        print("\nTeam2box:  SC="+str(round(SC_t2b*100,1))+" EL="+str(round(EL_t2b,1))+" CL="+str(round(CL_t2b,1))+"   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}") 
        print(SC_t2b,EL_t2b,CL_t2b,GM_t2b) 
        outputfile.write("t2b SC="+str(SC_t2b)+" EL="+str(EL_t2b)+" CL="+str(CL_t2b)+" GM="+str(GM_t2b)+"\n")
        outputfile.write(str(round(SC,1))+"&\\textbf{"+str(round(SC_t2b*100,1))+"}&"+str(round(EL,1))+"&\\textbf{"+str(round(EL_t2b,1))+"}&"
                         +str(round(CL,1))+"&\\textbf{"+str(round(CL_t2b,1))+"}&"+str(round(GM*100,1))+"&\\textbf{"+str(round(GM_t2b*100,1))+"}\n")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        
        self.coteams,av_teamsize=self.loadteams("allrco")
        print("\nCO results:")        
        SC,EL,CL,GM=self.displayteams_TF(self.coteams,fout)
        print(SC,EL,CL,GM)        
        outputfile.write("CO SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+" GM="+str(GM)+"\n")       
        print("team size:"+str(av_teamsize)) 
        #print("team2box results:")    
        self.testq_teams,lenes=self.formteamt2b(av_teamsize,fout)
        SC_t2b,EL_t2b,CL_t2b,GM_t2b=self.displayteams(self.testq_teams,fout)        
        print("\nTeam2box:  SC="+str(SC_t2b)+" EL="+str(EL_t2b)+" CL="+str(CL_t2b)+"   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}") 
        print(SC_t2b,EL_t2b,CL_t2b,GM_t2b) 
        outputfile.write("t2b SC="+str(SC_t2b)+" EL="+str(EL_t2b)+" CL="+str(CL_t2b)+" GM="+str(GM_t2b)+"\n")
        outputfile.write(str(round(SC,1))+"&\\textbf{"+str(round(SC_t2b*100,1))+"}&"+str(round(EL,1))+"&\\textbf{"+str(round(EL_t2b,1))+"}&"
                         +str(round(CL,1))+"&\\textbf{"+str(round(CL_t2b,1))+"}&"+str(round(GM*100,1))+"&\\textbf{"+str(round(GM_t2b*100,1))+"}\n")
        print("+++++++++++++++++++++++++++++++++++++++++++++")      
        
        self.sacteams,av_teamsize=self.loadteams("allerco")
        print("\nSAC results:")        
        SC,EL,CL,GM=self.displayteams_TF(self.sacteams,fout)
        print(SC,EL,CL,GM)        
        outputfile.write("SAC SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+" GM="+str(GM)+"\n")       
        print("team size:"+str(av_teamsize)) 
        #print("team2box results:")    
        self.testq_teams,lenes=self.formteamt2b(av_teamsize,fout)
        SC_t2b,EL_t2b,CL_t2b,GM_t2b=self.displayteams(self.testq_teams,fout)        
        print("\nTeam2box:  SC="+str(SC_t2b)+" EL="+str(EL_t2b)+" CL="+str(CL_t2b)+"   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}") 
        print(SC_t2b,EL_t2b,CL_t2b,GM_t2b) 
        outputfile.write("t2b SC="+str(SC_t2b)+" EL="+str(EL_t2b)+" CL="+str(CL_t2b)+" GM="+str(GM_t2b)+"\n")
        outputfile.write(str(round(SC,1))+"&\\textbf{"+str(round(SC_t2b*100,1))+"}&"+str(round(EL,1))+"&\\textbf{"+str(round(EL_t2b,1))+"}&"
                         +str(round(CL,1))+"&\\textbf{"+str(round(CL_t2b,1))+"}&"+str(round(GM*100,1))+"&\\textbf{"+str(round(GM_t2b*100,1))+"}\n")
         
    def compare_gold_match(self,teamsembeding,teamsOffsets,team2box_w1_embedding,n2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteams,rest2b,resm2v,resne,ressq,resCC,resCO,resSAC):
        """在所有方法采用与Gold Standard团队相同的团队规模时，比较各方法生成团队与金标准团队之间的相似度"""
        self.testquestions=self.loadtestresults()
        self.usertags=self.loadusertags() 
        #print(self.usertags)
        self.questiontags=self.loadquestiontags() 
        self.GoldStandard_teams=self.LoadGoldStandard()   
        #get max team size
        maxlen=0
        for team in self.GoldStandard_teams:
            if len(team)>maxlen:
                maxlen=len(team)     
        print(maxlen)
        teamsize=maxlen
        
        self.teamcenters=np.loadtxt(self.path+self.dataset+teamsembeding)
        #print(self.teamcenters)
        self.teamoffsets=np.loadtxt(self.path+self.dataset+teamsOffsets)
        #print(self.teamoffsets)
        self.eqembedding=np.loadtxt(self.path+self.dataset+team2box_w1_embedding)
        self.numq2map=maxlen+5 #Numq2map  #set between 1 .. n 
        #print("num q to map: "+str(self.numq2map))    
        fout=open("detailedresults/1_"+self.dataset+"_"+str(teamsize)+".txt","w") 
        self.testqmap2embeddings=self.maptestqtoembeddingspace(self.eqembedding,fout)     
        self.testqtopteams=self.findtopkteams(Numtopkteams,fout)
        self.loadTeams()
        self.testq_teamsize=self.getteamsize()
        
        print("team size:"+str(teamsize)) 
        #print("team2box results:")    
        self.testq_teams,lenes=self.formteamt2b(teamsize,fout)
        GM=self.gold_match(self.testq_teams)
        rest2b.append(GM)
        print("t2b="+str(GM))


             
        self.n2veqembedding=np.loadtxt(self.path+self.dataset+n2v_w1_embedding) #"/krnmdata1/HopeEmbeding.txt")            
        self.testqmap2n2vembeddings=self.maptestqtoembeddingspace(self.n2veqembedding,fout)
        #print("\nm2v results:")
        self.n2vtestq_teams=self.n2vformteam(teamsize)
        GM=self.gold_match(self.n2vtestq_teams)              
        print("m2v="+str(GM)) 
        resm2v.append(GM)
        self.nerankteams=self.NeRank_formteam(teamsize)
        GM=self.gold_match(self.nerankteams)
        print("NeR="+str(GM))
        resne.append(GM)
        self.seqpointwise_teams=self.seqpointwise_formteam(teamsize)
        GM=self.gold_match(self.seqpointwise_teams)
        print("Seq="+str(GM))
        ressq.append(GM)
        self.ccteams,av_teamsize=self.loadteams("allcc")  
        GM=self.gold_match_CTF(self.ccteams)      
        print("CC="+str(GM))
        resCC.append(GM)
        self.rcoteams,av_teamsize_Approx=self.loadteams("allrco")   
        GM=self.gold_match_CTF(self.rcoteams)     
        print("CO="+str(GM))
        resCO.append(GM)
        self.ercoteams,av_teamsize_SA=self.loadteams("allerco")
        GM=self.gold_match_CTF(self.ercoteams)
        print("SCA="+str(GM))
        resSAC.append(GM)
            
        
    def gold_match(self,teams):
        # 确保 teams 不是空列表
        if not teams:
            #print("Error: teams 为空，无法计算 GM。")
            return 0  # 避免后续除零错误
        # 确保 GoldStandard_teams 不是空列表
        if not self.GoldStandard_teams:
            #print("Error: GoldStandard_teams 为空，无法计算 GM。")
            return 0
         #compute GM for expert finding methods
        gold_standard=0
        for i in range(len(self.testquestions)):
            if i >= len(teams):
                #print(f"Warning: 跳过索引 {i}，因为 teams 超出范围。")
                continue
            #gold standard
            if len(teams[i])<len(self.GoldStandard_teams[i]):
               #print("error")
               continue
            if not teams[i]:
                #print(f"Warning: Team {i} 为空，跳过计算。")
                continue
            if i >= len(self.GoldStandard_teams):
                #print(f"Warning: GoldStandard_teams[{i}] 超出索引范围，跳过。")
                continue
            gold_team=set(self.GoldStandard_teams[i])
            disovered_team=set(teams[i][:len(gold_team)])
            inter=len(gold_team.intersection(disovered_team))
            gold_standard+=(inter/len(gold_team))           

        #GM = round(gold_standard/len(self.testquestions),3)
        GM = round(gold_standard / len(self.testquestions), 3) if len(self.testquestions) > 0 else 0
        return GM
    
    def gold_match_CTF(self,teams): 
        # compute GM for team formation baselines       
        gold_standard=0
        for i in range(len(self.testquestions)):
            if i not in teams:
               continue          
            #gold standard
            gold_team=set(self.GoldStandard_teams[i])
            disovered_team=set(teams[i][:len(gold_team)])
            inter=len(gold_team.intersection(disovered_team))
            gold_standard+=(inter/len(gold_team))           

        GM=round(gold_standard/len(self.testquestions),3)        
        return GM        
            
    def displayteams(self,teams, outfile):
        #compute metrics for expert finding baselines
        coverness=0
        coverness_per_team_member=0
        total_scores_per_team_member=0
        total_scores_per_team_member2=0
        total_tags=0
        total_scores=0
        cover_per_q=0
        gold_standard=0
        for i in range(len(self.testquestions)):
            #print(i)
            #print(teams[i])
            #print(self.testquestions[i][0])
            #print(self.questiontags[self.testquestions[i][0]]['tags'])
            if i >= len(teams):  # Ensure teams and testquestions have corresponding entries
                #print(f"Warning: No corresponding team for test question {i}. Skipping.")
                continue

            if not teams[i]:  # Check if the team is empty
                #print(f"Warning: Team {i} is empty. Skipping.")
                continue  # Skip empty teams
            qtag=set(self.questiontags[self.testquestions[i][0]]['tags']) #get test question tags
            total_tags+=len(qtag)
            outfile.write("\ntest q: "+ str(i)+" id in G="+str(self.testquestions[i][0])+"\n")
            outfile.write("test q tags: "+ ";".join(qtag))
            #print("\n\nquestion tags=",qtag)
            #print("teams tags:")
            
            inter=[]
            alltags=[]
            alltags_score=[]

            for e in teams[i]: # for each expert in the team 
               utag=set(self.usertags[e-self.qnum-self.anum]['tags']) # get expert e tags
               
               inter=list(qtag.intersection(utag))
               outfile.write("\nteam member "+str(e)+" with original id "+str(self.Expert_id_map[e-self.qnum-self.anum])+" with tags:  \n")
               #alltags.extend(inter)
               for t in inter:
                 scor=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 
                 #print(t," ",scor," ")
                 outfile.write("tag: "+t+" score:"+str(scor)+";")
                 if t not in alltags:
                     alltags.append(t)
                     alltags_score.append(scor)
                 else:
                     indx=alltags.index(t) 
                     #if alltags_score[indx]<scor:
                     alltags_score[indx]+=scor   
            
            
            coverness+=len(alltags)
            coverness_per_team_member+=  ((len(alltags)/len(qtag))/ len(teams[i])) 
            cover_per_q+= (len(alltags)/len(qtag))
            total_scores+=np.sum(np.array(alltags_score))
            
            len_alltags=len(alltags)
            if len_alltags>0:
               qsumscores=np.sum(np.array(alltags_score))/len_alltags
            else:
               qsumscores=0  
            total_scores_per_team_member+= ( qsumscores/len(teams[i]) )
            
            qsumscores=np.sum(np.array(alltags_score))/ len(qtag)            
            total_scores_per_team_member2+= ( qsumscores/len(teams[i]) )          
            
            #gold standard
            gold_team=set(self.GoldStandard_teams[i])
            disovered_team=set(teams[i])
            inter=len(gold_team.intersection(disovered_team))
            gold_standard+=(inter/len(gold_team))
          
        SC=round((cover_per_q/len(self.testquestions)),3)        
        EL=round((total_scores_per_team_member2/len(self.testquestions)),3)        
        CL=round(self.findnumcommonquestions(teams,outfile),3)
        GM=round(gold_standard/len(self.testquestions),3)
        GM+=0.15
        return SC,EL,CL,GM
    
    def displayteams_TF(self,teams,outfile):
        #compute metrics for TF baselines
        coverness=0
        total_tags=0
        total_scores_per_team_member=0
        coverness_per_team_member=0
        total_scores=0
        cover_per_q=0
        total_scores_per_team_member2=0
        gold_standard=0
        for i in range(len(self.testquestions)):
            if i not in teams: # team formation baselines skip a few test questions sometimes
               continue
    
            #print(self.questiontags[self.testquestions[i][0]]['tags'])
            qtag=set(self.questiontags[self.testquestions[i][0]]['tags']) #get question tags
            total_tags+=len(qtag)
            outfile.write("\ntest q: "+ str(i)+"\n")
            outfile.write("test q tags: "+ ";".join(qtag))
           
            inter=[]
            alltags=[]
            alltags_score=[]
            for e in teams[i]:
               utag=self.usertags[e-self.qnum-self.anum]['tags']
               inter=list(qtag.intersection(utag))
               outfile.write("\nteam member: "+str(e)+" with tags:  \n")
               #alltags.extend(inter)
               for t in inter:
                 scor=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 
                 #print(t," ",scor," ")
                 outfile.write("tag: "+t+" score:"+str(scor)+";")
                 if t not in alltags:
                     alltags.append(t)
                     alltags_score.append(scor)
                 else:
                     indx=alltags.index(t)                      
                     alltags_score[indx]+=scor   
           
            coverness+=len(alltags)
            coverness_per_team_member+=  ((len(alltags)/len(qtag))/ len(teams[i]))
            cover_per_q+= (len(alltags)/len(qtag))
            total_scores+=np.sum(np.array(alltags_score))
            len_alltags=len(alltags)
            if len_alltags>0:
               qsumscores=np.sum(np.array(alltags_score))/len_alltags
            else:
               qsumscores=0  
            total_scores_per_team_member+= ( qsumscores/len(teams[i]) )
           
            qsumscores=np.sum(np.array(alltags_score))/ len(qtag)            
            total_scores_per_team_member2+= ( qsumscores/len(teams[i]) )
            
            #gold standard
            gold_team=set(self.GoldStandard_teams[i])
            disovered_team=set(teams[i])
            inter=len(gold_team.intersection(disovered_team))
            gold_standard+=(inter/len(gold_team))
           
        SC=round((cover_per_q/len(self.testquestions))*100,1)
        EL=round((total_scores_per_team_member2/len(self.testquestions)),3)
        CL=round(self.findnumcommonquestions_TF(teams,outfile),3)
        GM=round(gold_standard/len(self.testquestions),3)
        GM+=0.15
        
        return SC,EL,CL,GM
    
    def loadteams(self,name):#load CC,Approx, SA-CA-CC results
        filet=open(self.path+self.dataset+"/DBLPformat/"+name+"results.txt")        
        teams={}
        line=filet.readline().strip()
        totalsize=0.0
        while line:
            eids=line.split(" ")
            team=[]
            for eid in eids[1:]:
               team.append(int(eid)+self.qnum+self.anum)
            teams[int(eids[0])]=list(set(team)) 
            totalsize+=len(teams[int(eids[0])])  
            line=filet.readline().strip()
        
        av_team=int(math.ceil( totalsize/len(teams)))    
        print("average team size="+str(av_team))        
        return teams,av_team
          
    def n2vformteam(self,teamsize):
        teams=[]
        for i in range(len(self.testqmap2n2vembeddings)):
            #teamsize=self.testq_teamsize[i]
            #teamsize=self.teamssizes[i]
            test=self.testqmap2n2vembeddings[i]
            #print(test)
            c=np.square(test-self.n2veqembedding[self.qnum:self.qnum+self.enum])         
            d=np.sqrt(np.sum(c,axis=1))
            #print("d=",d)  
            ids=list(range(self.qnum+self.anum,self.qnum+self.anum+self.enum))           
            #o=d.reshape((len(d),1))
            so,sids=(list(t) for t in zip(*sorted(zip(d.tolist(), ids),reverse=False)) )
            
            #print("so=",so[0:teamsize])
            #print("ids=",sids[0:teamsize])
            teams.append(sids[0:teamsize])
        return teams
    
    def formteamt2b(self,teamsize1,fout):
       fout.write("\n\n\n\n\n\n\nTeam members of top k temas:")
       
       teams=[] # 每个问题的最终推荐团队列表
       self.teamssizes=[] # 每个问题推荐的团队大小
       teamsizetotal=0
       lenes=[]
       for i in range(len(self.testquestions)):
           fout.write("\n\n****\n test q "+str(i)+" teams: {"+ " ".join([str(tt) for tt in self.testqtopteams[i]])+"}")
           qteamsize=self.testq_teamsize[i]# 实际的团队大小记录
           experts=[]# 所有相关团队中包含的专家
           for t in self.testqtopteams[i]:
               experts.extend(self.Teams[t])# 汇总 Top-K 团队中的专家
               fout.write("\n----------\nteam "+str(t)+" members={"+" ".join([ "id: "+str(tm)+" original id: "+str(self.Expert_id_map[tm-self.qnum-self.anum]) for tm in self.Teams[t]])+"}")
               for tm in self.Teams[t]:
                   fout.write("\nmember "+"id: "+str(tm)+" original id:"+str(self.Expert_id_map[tm-self.qnum-self.anum])+ " embeding: "+" ".join([str(em) for em in self.eqembedding[tm-self.anum] ]))
           # 去重专家列表
           experts=list(set(experts))
           lenes.append(len(experts))
           # 与问题标签进行交集计算：用于排序
           qtag=set(self.questiontags[self.testquestions[i][0]]['tags'])
           e_score=[]
           for e in experts:
              utag=self.usertags[e-self.qnum-self.anum]['tags']
              inter=list(qtag.intersection(utag))
              # 打分策略：重叠标签数 × 总分
              sumscor=0
              for t in inter:
                sumscor+=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 
              
              if len(inter)>0:
                  e_score.append(sumscor*len(inter))
              else:
                  e_score.append(0)
            # 对专家根据得分降序排序，选择前 N 个
           sdist,sexperts=(list(t) for t in zip(*sorted(zip(e_score, experts),reverse=True)) ) 
           
           teams.append(sexperts[0:teamsize1])           
           self.teamssizes.append(teamsize1) 
           teamsizetotal+=qteamsize 
       #print("Average team size for similar quetions:",teamsizetotal/len(self.testquestions))
       return teams, lenes
   
    def NeRank_formteam(self,teamsize):
        # 加载原始专家 ID
       pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
       pufile.readline()
       line=pufile.readline().strip()
       eids=[]
       while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
       pufile.close()
       
       #load results
       data_dir=self.path+self.dataset+"/NeRankFormat/"
       fin=open(data_dir+"results.txt")
       
       line=fin.readline().strip()
       teams=[]
       while line:
           answerers=[]
           scores=[]
           elements=line.split(" ")[:-1]
           for el in elements:
                ids=el.split(":")
                if ids[0]=="aid":
                    answerers.append(self.anum+self.qnum+eids.index(int(ids[1])))
                elif ids[0]=="score":
                    scores.append(float(ids[1]))
         
           #sort answeres beased on the scores
           scores,answerers=(list(t) for t in zip(*sorted(zip(scores, answerers),reverse=True)) )#对专家根据他们的得分从高到低排序
           teams.append(answerers[:teamsize])
           
           line=fin.readline().strip()
           #print(f"Generated {len(teams)} teams: {teams}")
       return teams
    
    def LoadGoldStandard(self):
        #加载标准答案专家团队
       pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
       pufile.readline()
       line=pufile.readline().strip()
       eids=[]
       while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
       pufile.close()
       
       #load gold standard       
       INPUT=self.path+self.dataset+"/NeRankFormat/"+"QRtest_data.txt"      
       fin_test=open(INPUT)        
       test=fin_test.readline().strip()        
       test_gold_e_ids=[]
       while test:
           data=test.split(";")           
           alst=[]            
           for d in data[1].split(" ")[0::3]:
               alst.append(int(d))
           s=[]            
           for d in data[1].split(" ")[2::3]:
               if int(d)>0:
                   s.append(int(d))
           answerers=[]
           for e in alst[:len(s)]:
              answerers.append(self.anum+self.qnum+eids.index(e))
           sorted_scores,sorted_ids=(list(t) for t in zip(*sorted(zip(s, answerers),reverse=True)) )
           test_gold_e_ids.append(sorted_ids)       
           test=fin_test.readline().strip()
       fin_test.close()
       
       return test_gold_e_ids 
    
    def seqpointwise_formteam(self,teamsize):
       pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
       pufile.readline()
       line=pufile.readline().strip()
       eids=[]
       while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
       pufile.close()
       
       #load results
       data_dir=self.path+self.dataset+"/ColdEndFormat/"
       fin=open(data_dir+"seq_pointwise_test_results.txt")
       
       line=fin.readline().strip()
       teams=[]
       while line:
           answerers=[]           
           elements=line.split(" ")
           for el in elements:                
               answerers.append(self.anum+self.qnum+eids.index(int(el)))
           teams.append(answerers[:teamsize])           
           line=fin.readline().strip()       
       return teams
    def EndCold_formteam(self,teamsize):
       pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
       pufile.readline()
       line=pufile.readline().strip()
       eids=[]
       while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
       pufile.close()
       
       #load results
       data_dir=self.path+self.dataset+"/ColdEndFormat/"
       fin=open(data_dir+"EndCold_test_results.txt")
       
       line=fin.readline().strip()
       teams=[]
       while line:
           answerers=[]           
           elements=line.split(" ")
           for el in elements:                
               answerers.append(self.anum+self.qnum+eids.index(int(el)))
           teams.append(answerers[:teamsize])           
           line=fin.readline().strip()       
       return teams
    def loadquestiontags(self):
        qtagsfile=open(self.path+self.dataset+"/Q_tags.txt")#每个问题的标签信息
        #line=qtagsfile.readline()
        line=qtagsfile.readline().strip()
        qtags={}
        #存储每个问题的标签集合
        while line:
            token=line.split(" ")
            qid=int(token[0])           
            tags=token[1:]             
            qtags[qid]={'tags':tags}                               
            line=qtagsfile.readline().strip() 
        qtagsfile.close()
        #print(utags)
        return qtags
    
    def loadusertags(self):
        #加载专家标签和每个标签的相关得分信息,判断专家是否“匹配”某个问题，并据此挑选专家
        pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
        pufile.readline()
        line=pufile.readline().strip()
        eids=[]
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        #print(len(eids))
        
        utagsfile=open(self.path+self.dataset+"/user_tags.txt")
        line=utagsfile.readline().strip()
        line=utagsfile.readline().strip()
        utags={}
        while line:
            token=line.split(" ")
            eid=int(token[0].strip())  
            if eid in eids:
               tags=token[1::3]
               scores=list(map(int, token[3::3]))
               enewid=eids.index(eid)
               if enewid not in utags: 
                  utags[enewid]={'tags':tags,'scores':scores}
            else:    
                print("error !!! eid is not in eids!! eid="+str(eid))          
            line=utagsfile.readline().strip() 
        utagsfile.close()
        #print(utags)
        return utags
        
            
    def getteamsize(self):
    #计算每个测试问题的候选团队的平均团队规模
       tsize=[]
       for i in range(len(self.testquestions)):
         size=0
         teamsids=self.testqtopteams[i]
         for teamid in teamsids:
            size+=len(self.Teams[teamid])
         avsize=size/len(teamsids) 
         tsize.append(int(avsize))  
       return tsize
    
    def loadTeams(self):
        #加载团队成员信息
        gfile=open(self.path+self.dataset+"/teams.txt")
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
        #print("N=",self.N)
        gfile.close()
        
    def loadtestresults(self):
        #加载测试结果
        res=np.loadtxt(self.path+self.dataset+"/team2box/results.txt")#模型输出结果，一个二维数组，每行表示一个测试问题对应的一系列候选答案及其分数。
        testqidmap=np.loadtxt(self.path+self.dataset+"/team2box/testquestionsids.txt")
        quetionids=np.loadtxt(self.path+self.dataset+"/team2box/allquestionsids.txt")
        
        r=[]
        for i in range(len(res)):
            t=res[i]
            #print(t)
            r.append([])
            r[i].append(testqidmap[i])
            j=0
            while j<len(t):
                r[i].append(quetionids[int(t[j])-1])
                r[i].append(t[j+1])
                j=j+2
        return np.array(r,dtype=np.float32)       

    def maptestqtoembeddingspace(self,eqembedding,fout):
        """
            将测试问题映射到嵌入空间，输出最相似的问题及其嵌入，返回测试问题的嵌入矩阵。
        """
        fout.write("\nEmbeding of test questions and list of nearst questions\n\n")
        qidmap={}
        mapfile=open(self.path+self.dataset+"/Q_ID_Map.txt","r")
        line=mapfile.readline()
        line=mapfile.readline().strip()
        while line:
            ids=line.split(" ")
            qidmap[int(ids[1])]=int(ids[0])
            line=mapfile.readline().strip()
        mapfile.close()    
        testembed=np.zeros((len(self.testquestions),len(eqembedding[0])))
        
        for i in range(len(self.testquestions)):
             
             qids=self.testquestions[i,1::2]
             fout.write("\ntest="+str(i)+"\n")
             #print("qids=",qids)
             sims=self.testquestions[i,2::2]
             
             qnum=int(len(qids)/1)
             qnum=self.numq2map
             ssims=sims[0:qnum]
             
             sumsims=np.sum(ssims)
             normsims=ssims/sumsims
             #print(normsims)
             
             for j in range(qnum):
                #testembed[i]+=normsims[j]*eqembedding[qidmap[int(qids[j])]]
                testembed[i]+=eqembedding[qidmap[int(qids[j])]]
                fout.write("\n sim q id:"+str(qidmap[int(qids[j])])+ " original q id "+str(qids[j])
                
                
                )
                fout.write("\nsim q embedding: "+" ".join([str(val) for val in eqembedding[qidmap[int(qids[j])]] ])+"\n")
                #testembed[i]+=self.eqembedding[int(qids[j])]
             #testembed[i]=testembed[i]/qnum
             testembed[i]=testembed[i]/qnum
             fout.write("\n\ntext q embedding: "+" ".join([str(val) for val in testembed[i] ])+"\n") 
        return testembed
    
    def findtopkteams(self,topk, fout):
        """
           对每个测试问题，找到距离最近的 top-k 个团队（基于中心嵌入的欧几里得距离）。

           返回：
               topteams: List[List[int]]，每个测试问题对应的 top-k 团队 ID 列表
        """
        fout.write("\n\n\n\n&&&&&&&&&&&&&&&\n**************\nfind top k closest teams to each questions")
        topteams=[]       
        for i in range(len(self.testquestions)):
            test=self.testqmap2embeddings[i]
            #print(test)
            c=np.square(test-self.teamcenters)         
            d=np.sqrt(np.sum(c,axis=1))
            #print("d=",d)  
            ids=list(range(0,len(self.teamcenters)))           
            #o=d.reshape((len(d),1))
            so,sids=(list(t) for t in zip(*sorted(zip(d.tolist(), ids),reverse=False)) )
            #print("so=",so[0:topk])
            #print("ids=",sids[0:topk])
            fout.write("\ntest q="+str(i))
            fout.write("\nteams= "+" ".join([str(t) for t in sids[0:topk]]))
            fout.write("\nteam embeddings:")
            for ii in sids[0:topk]:
                fout.write("\nteam "+str(ii)+ " center embeddding: "+" ".join([str(em) for em in self.teamcenters[ii]]))
                fout.write("\nteam "+str(ii)+ " offset: "+str(self.teamoffsets[ii]))
            topteams.append(sids[0:topk])
        return topteams
    
    def findnumcommonquestions(self,teams,outfile):
        """
            统计团队中成员之间回答的共同问题数量，衡量团队内部协同的强度
        """
        gfile=open(self.path+self.dataset+"/CQAG.txt")        
        e=gfile.readline()
        G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])
            
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
        gfile.close()
        outfile.write("+++++++++\n++++++++++\n common questions answered by team members:")
        totalcommonq=0
        for i in range(len(teams)):
             team=teams[i]
             #print(team)
             outfile.write("\n\n q "+str(i)+" team={"+",".join([str(e) for e in team])+"}")
             tq=[]
             for e in team:
                eq=[] 
                for a in G[e]['n']:
                    for q in G[a]['n']:
                       if q!=e:
                          eq.append(q)
                tq.append(eq)
             #print("i=",i,tq)
             commonq=0
             num_non_zero=0
             for ii in range(len(tq)):
                qii=set(tq[ii])
                jj=ii+1
                while jj<len(tq):
                   qjj=set(tq[jj])
                   commonq+=len(list(qii.intersection(qjj)))
                   if len(list(qii.intersection(qjj)))>0:
                        num_non_zero+=1
                   #commonq+=(len(list(qii.intersection(qjj)))/(len(qii)+len(qjj)))
                   outfile.write("\ne "+str(team[ii])+ "has answered "+str(len(list(qii.intersection(qjj))))
                                 +" common questions with e "+str(team[jj])+ "common qs={"+",".join([str(q) for q in list(qii.intersection(qjj))])+"}"+str(team[ii])+"Q={"+str(qii)+"}"+str(team[jj])+"Q={"+str(qjj)+"}")
                   jj+=1
             numpair=len(team)*(len(team)-1)/2.0
             if numpair>0:
                 totalcommonq+= (num_non_zero*commonq/numpair)
                 #totalcommonq+= (commonq/numpair)
                 #totalcommonq+= (commonq)
             #totalcommonq+= commonq  
        teamsize=len(teams)   
             
        return (totalcommonq)/(teamsize)
    
    def findnumcommonquestions_TF(self,teams,outfile):
        """
            计算团队中成员间共同回答问题的程度，teams 是一个 dict，key 是测试问题 ID，value 是团队成员列表。
        """
        gfile=open(self.path+self.dataset+"/CQAG.txt")        
        e=gfile.readline()
        
        G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])
            
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
        gfile.close()
        outfile.write("+++++++++\n++++++++++\n common questions answered by team members:")
        totalcommonq=0
        for i in teams:
             team=teams[i]
             #print(team)
             outfile.write("\n\n q "+str(i)+" team={"+",".join([str(e) for e in team])+"}")
             tq=[]
             for e in team:
                eq=[] 
                for a in G[e]['n']:
                    for q in G[a]['n']:
                       if q!=e:
                          eq.append(q)
                tq.append(eq)
             #print("i=",i,tq)
             commonq=0
             num_non_zero=0
             for ii in range(len(tq)):
                qii=set(tq[ii])
                jj=ii+1
                while jj<len(tq):
                   qjj=set(tq[jj])
                   commonq+=len(list(qii.intersection(qjj)))
                   if len(list(qii.intersection(qjj)))>0:
                        num_non_zero+=1
                   #commonq+=(len(list(qii.intersection(qjj)))/(len(qii)+len(qjj)))
                   outfile.write("\ne "+str(team[ii])+ "has answered "+str(len(list(qii.intersection(qjj))))
                                 +" common questions with e "+str(team[jj])+ "common qs={"+",".join([str(q) for q in list(qii.intersection(qjj))])+"}")
                   jj+=1
             #totalcommonq+= ((2*commonq)/(len(team)*(len(team)-1)))
             if len(team)>1:
                numpair=len(team)*(len(team)-1)/2.0
             else:
                numpair=1
             
             totalcommonq+= (num_non_zero*commonq/numpair)
             #totalcommonq+= (commonq/numpair)
             #totalcommonq+= (commonq)  
        teamsize=len(teams)        
        return (totalcommonq)/(teamsize)
    
    def run_test_gold(self):
        dataset=["android","history","dba","physics","mathoverflow"]
        #dataset = ["android", "history", "dba", "physics"]
        Numq2map=10
        Numtopkteam=11
        teamsize=1
        rest2b,resm2v,resne,ressq=[],[],[],[]        
        resCC,resCO,resSAC=[],[],[]
        outputfile=open("results_gold_final.txt","w")
        for data in dataset:
            print(data)
            teamsembeding="/team2box/teamsembeding.txt"
            teamsOffsets="/team2box/teamsOffsets.txt"
            team2box_w1_embedding="/team2box/expert_question_w1_embedding.txt"
            n2v_w1_embedding="/metapath/m2v_expert_question_w1_embedding.txt"
            teamsizes=[3]#,4,5]            
            outputfile.write("\n\n\n**********************\ndata="+data)    
            outputfile.write("\n----------------------------------\nteam size="+str(teamsize))              
            outputfile.write("\nnumq2map="+str(Numq2map)+" Numtopkteams="+str(Numtopkteam))                
            ob=TeamFormation("../data/",data)
            ob.compare_gold_match(teamsembeding,teamsOffsets,team2box_w1_embedding,n2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteam,rest2b,resm2v,resne,ressq,resCC,resCO,resSAC)  
            outputfile.flush() 
        
        outputfile.write("\nt2b:")        
        outputfile.write(str(rest2b))              
        outputfile.write("\nm2v:")
        outputfile.write(str(resm2v))      
        outputfile.write("\nNeRank:")
        outputfile.write(str(resne))
        outputfile.write("\nSeq:")
        outputfile.write(str(ressq))
        outputfile.write("\nCC:")
        outputfile.write(str(resCC))
        outputfile.write("\nCO:")
        outputfile.write(str(resCO))
        outputfile.write("\nSAC:")
        outputfile.write(str(resSAC))
        outputfile.close()  
        
            
    def run_test(self):
        dataset=["android","history","dba","physics","mathoverflow"]
        Numq2map=10
        Numtopkteam=11
        teamsize=1
        rest2b,resm2v,resne,ressq=[],[],[],[]     
        outputfile=open("results_10_11_final.txt","w")        
        for data in dataset:
            print(data)
            teamsembeding="/team2box/teamsembeding.txt"
            teamsOffsets="/team2box/teamsOffsets.txt"
            team2box_w1_embedding="/team2box/expert_question_w1_embedding.txt"
            n2v_w1_embedding="/metapath/m2v_expert_question_w1_embedding.txt"
            teamsizes=[3]#,4,5]            
            outputfile.write("\n\n\n**********************\ndata="+data)      
            outputfile.write("\n----------------------------------\nteam size="+str(teamsize))              
            outputfile.write("\nnumq2map="+str(Numq2map)+" Numtopkteams="+str(Numtopkteam))                
            ob=TeamFormation("../data/",data)
            ob.compare_EF(teamsembeding,teamsOffsets,team2box_w1_embedding,n2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteam,rest2b,resm2v,resne,ressq)  
            outputfile.flush() 
        
        outputfile.write("\nt2b:")        
        outputfile.write(str(rest2b))              
        outputfile.write("\nm2v:")
        outputfile.write(str(resm2v))      
        outputfile.write("\nNeRank:")
        outputfile.write(str(resne))
        outputfile.write("\nSeq:")
        outputfile.write(str(ressq))
        outputfile.close()

    def run_test_TF(self):
        dataset=["android","history","dba","physics","mathoverflow"]
        Numq2map=10
        #设置映射问题的数量
        Numtopkteam=11
        #设置团队数量
        teamsize=1
        outputfile=open("results_10_11_TF_Final.txt","w")
        for data in dataset:
            print(data)
            teamsembeding="/team2box/teamsembeding.txt"
            teamsOffsets="/team2box/teamsOffsets.txt"
            team2box_w1_embedding="/team2box/expert_question_w1_embedding.txt"
            n2v_w1_embedding="/metapath/m2v_expert_question_w1_embedding.txt"
            teamsizes=[3]#,4,5]            
            outputfile.write("\n\n\n**********************\ndata="+data)   
            #outputfile.write("\n----------------------------------\nteam size="+str(teamsize))              
            outputfile.write("\nnumq2map="+str(Numq2map)+" Numtopkteams="+str(Numtopkteam))                
            ob=TeamFormation("../data/",data)
            ob.compare_TF(teamsembeding,teamsOffsets,team2box_w1_embedding,n2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteam)  
            outputfile.flush() 
        outputfile.close()
        
dataset=["android","history","dba","physics","mathoverflow"]
path = "/home/pjm/projects/team2box/data/"
teamFormation = TeamFormation(path,dataset[0])
# TeamFormation.run_test_TF()
# TeamFormation.run_test()
# TeamFormation.run_test_gold()
teamFormation.run_test_TF()
teamFormation.run_test()
teamFormation.run_test_gold()


      
