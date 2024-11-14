import numpy as np
# 第三方进度条库
from tqdm import tqdm


class HMM:
    def __init__(self):
        # 标记-id
        self.tag2id = {'E-EDU': 0,
                       'B-RACE': 1,
                       'E-TITLE': 2,
                       'B-NAME': 3,
                       'M-NAME': 4,
                       'M-CONT': 5,
                       'M-ORG': 6,
                       'B-CONT': 7,
                       'B-EDU': 8,
                       'B-LOC': 9,
                       'B-ORG': 10,
                       'B-TITLE': 11,
                       'E-CONT': 12,
                       'E-ORG': 13,
                       'E-NAME': 14,
                       'M-TITLE': 15,
                       'E-LOC': 16,
                       'B-PRO': 17,
                       'M-LOC': 18,
                       'O': 19,
                       'M-PRO': 20,
                       'M-EDU': 21,
                       'E-PRO': 22,
                       'E-RACE': 23,
                       'S-RACE': 24,
                       'S-NAME': 25,
                       'M-RACE': 26,
                       'S-ORG': 27
                       }
        # id-标记
        self.id2tag = dict(zip(self.tag2id.values(), self.tag2id.keys()))
        # 表示所有可能的标签个数N
        self.num_tag = len(self.tag2id)
        # 所有字符的Unicode编码个数 x16
        self.num_char = 65535
        # 转移概率矩阵,N*N
        self.A = np.zeros((self.num_tag, self.num_tag))
        # 发射概率矩阵,N*M
        self.B = np.zeros((self.num_tag, self.num_char))
        # 初始隐状态概率,N
        self.pi = np.zeros(self.num_tag)
        # 无穷小量
        self.epsilon = 1e-100

    def train(self, corpus_path):
        '''
        函数功能：通过数据训练得到A、B、pi
        :param corpus_path: 数据集文件路径
        :return: 无返回值
        '''
        with open(corpus_path, mode='r', encoding='utf-8') as f:
            # 读取训练数据
            lines = f.readlines()
        print('开始训练数据：')
        for i in tqdm(range(len(lines))):
            if len(lines[i]) == 1:
                # 空行，即只有一个换行符，跳过
                continue
            else:
                # split()的时候，多个空格当成一个空格
                cut_char, cut_tag = lines[i].split()
                # ord是python内置函数
                # ord(c)返回字符c对应的十进制整数
                self.B[self.tag2id[cut_tag]][ord(cut_char)] += 1
                if len(lines[i - 1]) == 1:
                    # 如果上一个数据是空格
                    # 即当前为一句话的开头
                    # 即初始状态
                    self.pi[self.tag2id[cut_tag]] += 1
                    continue
                pre_char, pre_tag = lines[i - 1].split()
                self.A[self.tag2id[pre_tag]][self.tag2id[cut_tag]] += 1
        # 为矩阵中所有是0的元素赋值为epsilon
        self.pi[self.pi == 0] = self.epsilon
        # 防止数据下溢,对数据进行对数归一化
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))
        self.A[self.A == 0] = self.epsilon
        # axis=1将每一行的元素相加，keepdims=True保持其二维性
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.epsilon
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        print('训练完毕！')

    def viterbi(self, Obs):
        """
        函数功能：使用viterbi算法进行解码
        :param Obs: 要解码的中文String
        :return: 预测的tagList
        """
        # 获得观测序列的文本长度
        T = len(Obs)
        # T*N
        delta = np.zeros((T, self.num_tag))
        # T*N
        psi = np.zeros((T, self.num_tag))
        # ord是python内置函数
        # ord(c)返回字符c对应的十进制整数
        # 初始化
        delta[0] = self.pi[:] + self.B[:, ord(Obs[0])]
        # range（）左闭右开
        for i in range(1, T):
            # arr.reshape(4,-1) 将arr变成4行的格式，列数自动计算的(c=4, d=16/4=4)
            temp = delta[i - 1].reshape(self.num_tag, -1) + self.A
            # 按列取最大值
            delta[i] = np.max(temp, axis=0)
            # 得到delta值
            delta[i] = delta[i, :] + self.B[:, ord(Obs[i])]
            # 取出元素最大值对应的索引
            psi[i] = np.argmax(temp, axis=0)
        # 最优路径回溯
        path = np.zeros(T)
        path[T - 1] = np.argmax(delta[T - 1])
        for i in range(T - 2, -1, -1):
            path[i] = int(psi[i + 1][int(path[i + 1])])

        tagList = []
        for i in range(len(path)):
            tagList.append(self.id2tag[path[i]])
        return tagList

    def calculate(self, TrueTagList, PredictedTagList, outFile):
        answer = []
        # 分别计算每种tag的准确率和召回率
        for tag in self.tag2id.keys():
            # 计算准确率
            denominator = 1e-6
            Numerator = 1e-6
            for i in range(len(PredictedTagList)):
                for j in range(len(PredictedTagList[i])):
                    if PredictedTagList[i][j] == tag:
                        denominator += 1
                        if TrueTagList[i][j] == tag:
                            Numerator += 1
            p = Numerator/denominator
            # 计算召回率
            denominator2 = 1e-6
            Numerator2 = 1e-6
            for i in range(len(TrueTagList)):
                for j in range(len(TrueTagList[i])):
                    if TrueTagList[i][j] == tag:
                        denominator2 += 1
                        if PredictedTagList[i][j] == tag:
                            Numerator2 += 1
            r = Numerator2/denominator2
            # 构建输出字符串
            string = tag+": "+"准确率： "+str(p)+"召回率： "+str(r)
            answer.append(string)

        with open(outFile, 'w', encoding='utf-8') as f:
            for i in range(len(answer)):
                f.write(answer[i]+"\n")
