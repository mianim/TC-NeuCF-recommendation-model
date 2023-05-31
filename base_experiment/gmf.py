import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import os

from src.main import load_ncf_data

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

'''
train_dataset = NCFData(train_data, item_num, train_mat, num_ng=4, is_training=True)  # neg_items=4,default

'''


class NCFData(torch.utils.data.Dataset):  # define the dataset
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        '''
        features:下面的函数中传入的是train_data [userid,item1,item2.。。。。]
        num_item:物品的数量
        train_mat:用户-物品-（0,1）
        num_ng:负采样物品
        '''
        super(NCFData, self).__init__()
        # 标签仅在训练师有用
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    # 负采样的选择函数
    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        '''features_ng 负采样的特征列表'''
        self.features_ng = []
        # features_ps[user item1 item2]
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):  # 负采样的数量
                '''
                j:在商品的数量中随机选出一个数字
                '''
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])
        '''
        正采样就是训练集中有的物品
        负采样就是训练集中没有的物品
        features_ng[user,no_item1,no_item2....]
        '''
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]
        '''
        features_fill:[(user,item1),
        (user,item2),no_item1,no_item2]
        '''
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        '''
        if self.is_training:
            self.ng_sample()
            features = self.features_fill
            labels = self.labels_fill
        else:
            features = self.features_ps
            labels = self.labels
        '''
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels
        '''
        training :
            user
        '''
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


class GMF(nn.Module):
    def __init__(self, num_user, num_item, factor_num):
        super(GMF, self).__init__()
        '''
        用户Embedding层
        物品Embedding层
        线性层
        初始化权重

        '''
        self.embed_user_GMF = nn.Embedding(num_user, factor_num)
        self.embed_item_GMF = nn.Embedding(num_item, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_wieght_()

    def _init_wieght_(self):
        # 将权重初始化成高斯分布 userEmbedding有一个weight
        # itemEmbedding有一个weight 所以要初始化两个weight
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)

    def forward(self, user, item):
        '''
        将用户 和 物品输入到对应的embedding 层
        然后将两个embedding 进行内积
        将内积的结果输入到预测层（线性层进行预测）
        最后输出
        '''
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output = embed_user_GMF * embed_item_GMF
        predict = self.predict_layer(output)
        return predict.view(-1)


# 加载数据
def load_dataset(test_num=100):
    train_data = pd.read_csv("data/ratings.csv",
                             sep=',', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    print("训练数据：",train_data)
    '''在user_num的最大值处+1 得到用户的数量'''
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    # 将train_data转换为list形式[[user,item],[user,item]]
    train_data = train_data.values.tolist()

    # 初始化train_mat为稀疏矩阵的形式 里面存储的是 （如果用户对电影有评分则该处置1 反之为0）
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    '''
    train_mat的形式是(user,item):1
    '''
    test_data = []
    with open("D:\\pythonProject\\newPytorch\\矩阵分解\\Data\\ml-1m.test.negative", 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            # eval()
            # 函数用来执行一个字符串表达式，并返回表达式的值。还可以把字符串转化为list、tuple、dict。
            '''
            ml-1m.test.negative的形式为(user, positive_item)。。。。(后面是99个负采样) 
            '''
            u = eval(arr[0])[0]  # 去除user
            test_data.append([u, eval(arr[0])[1]])  # one postive item
            for i in arr[1:]:
                test_data.append([u, int(i)])  # 99 negative items
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


# 评估函数
def hit(gt_item, pre_item):
    if gt_item in pre_item:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # using gpu
cudnn.benchmark = True
# 封装train_dataset test_dataset train_loader(可迭代的数据类型方便后期的训练)
train_data, test_data, user_num, item_num, train_mat = load_ncf_data()


train_dataset = NCFData(train_data, item_num, train_mat, num_ng=4, is_training=True)  # neg_items=4,default
'''
train_dataset:return user,item,abel
'''
test_dataset = NCFData(test_data, item_num, train_mat, num_ng=0, is_training=False)  # 100

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
# every user have 99 negative items and one positive items，so batch_size=100 每一个用户有99个负样本 一个正样本 所以batch_size是100
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=99 + 1, shuffle=False, num_workers=2)
# training and evaluationg
# Setting GPU Enviroment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # using gpu
cudnn.benchmark = True
# training and evaluationg
print("%3s%20s%20s%20s" % ('K', 'Iterations', 'HR', 'NDCG'))
# 开始训练
for k in [8, 16, 32, 64]:
    model = GMF(int(user_num), int(item_num), factor_num=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    crition = nn.BCEWithLogitsLoss()
    best_hr, best_ndcg = 0.0, 0.0
    if __name__ == '__main__':
        for epoch in range(20):
            model.train()
            train_loader.dataset.ng_sample()
            for user, item, label in train_loader:
                user = user.to(device)
                item = item.to(device)
                label = label.float().to(device)

                # 梯度清零
                model.zero_grad()
                # 将要预测的user item 扔到模型里面 预测出来的和label进行比较
                prediction = model(user, item)
                loss = crition(prediction, label)
                loss.backward()
                optimizer.step()
            model.eval()
            HR, NDCG = metrics(model, test_loader, top_k=10)
            print("HR: {:.3f}\tNDCG: {:.3f}".format(HR, NDCG))
            if HR > best_hr: best_hr = HR
            if NDCG > best_ndcg: best_ndcg = NDCG
        print("%3d%20d%20.6f%20.6f" % (k, 20, best_hr, best_ndcg))











