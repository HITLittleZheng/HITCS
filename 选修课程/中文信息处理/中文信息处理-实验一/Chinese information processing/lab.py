#!/usr/bin/env python3
# coding=utf-8
 
import urllib
import re
import random
import string
import operator
'''
实现了 NGram 算法，同时通过正向和逆向最大匹配中选着概率更大的一个
'''
# class ScoreInfo:
#     score = 0
#     content = ''
 
class NGram:
    __dicWordFrequency = dict() #词频
    __dicPhraseFrequency = dict() #词段频
    __dicPhraseProbability = dict() #词段概率
 
    def printNGram(self):#打印词典
        print('词频')
        for key in self.__dicWordFrequency.keys():
            print('%s\t%s'%(key,self.__dicWordFrequency[key]))
        print('词段频')
        for key in self.__dicPhraseFrequency.keys():
            print('%s\t%s'%(key,self.__dicPhraseFrequency[key]))
        print('词段概率')
        for key in self.__dicPhraseProbability.keys():
            print('%s\t%s'%(key,self.__dicPhraseProbability[key]))
 
    def append(self,content):#统计词频
        '''
        训练 ngram  模型
        :param content: 训练内容
        :return: 
        '''
        #clear
        content = re.sub('\s|\n|\t','',content)
        ie = self.getIterator(content) #2-Gram 模型
        keys = []
        for w in ie:
            #词频
            k1 = w[0]
            k2 = w[1]
            if k1 not in self.__dicWordFrequency.keys():
                self.__dicWordFrequency[k1] = 0
            if k2 not in self.__dicWordFrequency.keys():
                self.__dicWordFrequency[k2] = 0
            self.__dicWordFrequency[k1] += 1
            self.__dicWordFrequency[k2] += 1
            #词段频
            key = '%s%s'%(w[0],w[1])
            keys.append(key)
            if key not in self.__dicPhraseFrequency.keys():
                self.__dicPhraseFrequency[key] = 0
            self.__dicPhraseFrequency[key] += 1
 
        #词段概率
        for w1w2 in keys:
            w1 = w1w2[0]
            w1Freq = self.__dicWordFrequency[w1]
            w1w2Freq = self.__dicPhraseFrequency[w1w2]
            # P(w1w2|w1) = w1w2出现的总次数/w1出现的总次数 = 827/2533 ≈0.33 , 即 w2 在 w1 后面的概率
            self.__dicPhraseProbability[w1w2] = round(w1w2Freq/w1Freq,2)
        pass
 
    def getIterator(self,txt):#迭代器
        '''
        bigram 模型迭代器
        :param txt: 一段话或一个句子
        :return: 返回迭代器，item 为 tuple，每项 2 个值
        '''
        ct = len(txt)
        if ct<2:
            return txt
        for i in range(ct-1):
            w1 = txt[i]
            w2 = txt[i+1]
            yield (w1,w2)
 
    def getScore(self,txt): #从正向匹配算法和逆向最大匹配算法的结果中选一个
        '''
        使用 ugram 模型计算 str 得分
        :param txt: 
        :return: 
        '''
        # ie = self.getIterator(txt)
        # score = 1
        # fs = []
        # for w in ie:
        #     key = '%s%s'%(w[0],w[1])
        #     freq = self.__dicPhraseProbability[key]
        #     fs.append(freq)
        #     score = freq * score
        # #print(fs)
        # #return str(round(score,2))
        # info = ScoreInfo()
        # info.score = score
        # info.content = txt
        # return infore
        dic1,dic2 = self.extract_top_half()
        result1 = cut_words(txt,dic2)
        result2 = cut_words_reverse(txt,dic2)
        score1,score2 = 1,1
        for w in result1:
            if w in self.__dicWordFrequency:
                freq = self.__dicWordFrequency[w]
            elif w in self.__dicPhraseProbability:
                freq = self.__dicPhraseProbability[w]
            else:
                score1 += 1
                freq = 1
            score1 = score1 * freq
        for w in result2:
            if w in self.__dicWordFrequency:
                freq = self.__dicWordFrequency[w]
            elif w in self.__dicPhraseProbability:
                freq = self.__dicPhraseProbability[w]
            else:
                score2 += 1
                freq = 1
            score2 = score2 * freq
        if score1 >= score2:
            return result1
        else:
            return result2

    def extract_top_half(self): #挑选前一半的两字词作为正常词
        dicPhraseFrequency = sorted(self.__dicPhraseFrequency.items(), key=lambda x: x[1], reverse=True)
        top_half_items1 = dicPhraseFrequency[:len(dicPhraseFrequency) // 2]
        dicPhraseProbability = sorted(self.__dicPhraseProbability.items(), key=lambda x: x[1], reverse=True)
        top_half_items2 = dicPhraseProbability[:len(dicPhraseProbability) // 2]

        result_dict1 = {item[0]: item[1] for item in top_half_items1}
        result_dict2 = {item[0]: item[1] for item in top_half_items2}

        return result_dict1,result_dict2
 
def fileReader(): #读取训练集
    path = "data_re.txt"
    text = []
    with open(path,'r',encoding='utf-8') as f:
        rows = 0
        # 按行统计
        while True:
            rows += 1
            line = f.readline()
            text.append(line)
            if not line:
                print('训练集读取结束 %s'%path)
                return text
            print('content rows=%s len=%s type=%s'%(rows,len(line),type(line)))
    pass
 
def getData(): #读取测试集
    text_path = 'text.txt'
    arr = []
    with open(text_path,'r',encoding='utf-8') as f:
        while True:
            juzi = f.readline()
            arr.append(juzi)
            if not juzi:
                print("测试集读取结束")
                return arr
    
 
#  实现正向匹配算法中的切词方法
def cut_words(raw_sentence, words_dic):
    '''
    :param raw_sentence: 需要分词句子
    :param words_dic: 词典列表
    :return: 
    '''
    max_length = max(len(word) for word in words_dic) #  统计词典中最长的词
    sentence = raw_sentence.strip()
    words_length = len(sentence) # 统计序列长度
    cut_word_list = []   # 存储切分好的词语
    while words_length > 0:
        max_cut_length = min(max_length, words_length)
        subSentence = sentence[0: max_cut_length]
        while max_cut_length > 0:
            if subSentence in words_dic:
                cut_word_list.append(subSentence)
                break
            elif max_cut_length == 1:
                cut_word_list.append(subSentence)
                break
            else:
                max_cut_length = max_cut_length -1
                subSentence = subSentence[0:max_cut_length]
        sentence = sentence[max_cut_length:]
        words_length = words_length - max_cut_length
    return cut_word_list
 
# 实现逆向最大匹配算法中的切词方法
def cut_words_reverse(raw_sentence, words_dic):
    max_length = max(len(word) for word in words_dic) # 统计词典中词的最长长度
    sentence = raw_sentence.strip()
    words_length = len(sentence)# 统计序列长度
    cut_word_list = []# 存储切分出来的词语
    # 判断是否需要继续切词
    while words_length > 0:
        max_cut_length = min(max_length, words_length)
        subSentence = sentence[-max_cut_length:]
        while max_cut_length > 0:
            if subSentence in words_dic:
                cut_word_list.append(subSentence)
                break
            elif max_cut_length == 1:
                cut_word_list.append(subSentence)
                break
            else:
                max_cut_length = max_cut_length -1
                subSentence = subSentence[-max_cut_length:]
        sentence = sentence[0:-max_cut_length]
        words_length = words_length - max_cut_length
    cut_word_list.reverse()
    return cut_word_list

def main():  
    ng = NGram()
    reader = fileReader()
    print("训练集个数：",len(reader))
    #将语料追加到 bigram 模型中
    for row in reader:
        ng.append(row)
    #ng.printNGram()
    #测试生成的句子，是否合理
    arr = getData()
    print("测试集个数：",len(arr))
    print(len(ng._NGram__dicPhraseFrequency))
    print(len(ng._NGram__dicPhraseProbability))
    
    
    infos= []
    for s in arr:
        info = ng.getScore(s)
        infos.append(info)
    result_path = 'result.txt'
    with open(result_path,'w',encoding='utf-8') as f:
        for i in infos:
            for j in i:
                f.write(j+' ')
            f.write('\n')
    pass

if __name__ == '__main__':
    main()
    pass

