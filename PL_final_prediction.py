import nltk as nl
from nltk.tokenize import word_tokenize
from PL import predict

teamstats = open("E:/Programming/NLP/PL/stats.txt" , "r" ).readlines()
stats = [word_tokenize(line) for line in teamstats]
RTW_data = [(e[0] , [float(e[1]),int(e[2])]) for e in stats]
dic = {}
for key , value in RTW_data:
    dic[key] = value
fixtures = [word_tokenize(li) for li in open("E:/Programming/NLP/PL/upcoming.txt" , "r" ).readlines()] 
res = open("E:/Programming/NLP/PL/results.txt" , "a")
res.write("\n================== NEW PREDICTIONS ==================\n\n")
for f in fixtures:
    AI_predict = predict([dic[f[0]][0] , dic[f[1]][0]])
    if AI_predict >  0.491 and AI_predict < 0.511:
        dic[f[0]][1]+=1
        dic[f[1]][1]+=1
        if (dic[f[0]][0] - 0.125) >= 0:
            dic[f[0]][0]-=0.125
            if (dic[f[0]][0] <1.5):
                dic[f[0]][0]+=0.125
            else:
                dic[f[1]][0]-=0.125
        if (dic[f[1]][0]-0.125 >=0):
            if (dic[f[1]][0] <1.5):
                dic[f[1]][0]+=0.125
            else:
                dic[f[1]][0]-=0.125
        res.write(f[0] +" "+ f[1] + " DRAW\n")
        
    elif AI_predict<0.49:
        dic[f[0]][1]+=3
        if (dic[f[0]][0]+0.25<=3): 
            dic[f[0]][0]+=0.25
        res.write(f[0] + " BEAT "+ f[1]+"\n")
    elif AI_predict>0.51:
        dic[f[1]][1]+=3
        if (dic[f[1]][0]+0.25 <=3):
            dic[f[1]][0]+=0.25
        res.write(f[0] + " LOST AGANIST "+ f[1]+"\n")
stand = open("E:/Programming/NLP/PL/finalstand.txt" , "w")
for key in dic:
    stand.write(key + "\t" + str(dic[key][0]) + "\t" + str(dic[key][1]))
    stand.write("\n")
