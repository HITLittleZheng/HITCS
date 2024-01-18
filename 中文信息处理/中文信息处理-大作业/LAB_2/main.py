import HMM
import buildcorpus as bc


def UpdateFile(inPath, outPath):
    wordList = []
    with open(inPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        N = len(lines)
        wordList.append(lines[0])
        for i in range(N-1):
            word = lines[i+1]
            wordList.append(word)
            if wordList[-1] == '\n' and ("，" in wordList[-2]):
                # 去掉多余的换行符号
                wordList.pop()
    with open(outPath, 'w', encoding='utf-8') as f2:
        for i in range(len(wordList)):
            f2.write(wordList[i])


def out(wordList, predictedTagList, outPath):
    N = len(wordList)
    for i in range(N):
        for j in range(len(wordList[i])):
            # 处理wordlist
            wordList[i][j] += " "+predictedTagList[i][j]

    with open(outPath, 'w', encoding='utf-8') as f:
        for m in range(N):
            for n in range(len(wordList[m])):
                f.write(wordList[m][n]+"\n")
            f.write("\n")


UpdateFile("./txt/train.txt","./txt/newTrain.txt")
UpdateFile("./txt/test.txt","./txt/newTest.txt")
hmm = HMM.HMM()
hmm.train("./txt/newTrain.txt")
TrueWordList, TrueTagList = bc.build_corpus("./txt/newTest.txt")

predictTagList = []
# 对每个句子进行解码，求出该句子预测的tag
for x in TrueWordList:
    tag = hmm.viterbi(x)
    # 加入预测的tagList中
    predictTagList.append(tag)
out(TrueWordList, predictTagList, "./txt/out.txt")
hmm.calculate(TrueTagList, predictTagList,"./txt/evaluate.txt")