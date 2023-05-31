import pandas as pd
import torch as pt
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

BATCH_SIZE=100

# 读取测试以及训练数据
cols=['user','item','rating','timestamp']
train=pd.read_csv('data/ratings_train.csv', encoding='utf-8', names=cols)
test=pd.read_csv('data/ratings_test.csv', encoding='utf-8', names=cols)

# 去掉时间戳
train=train.drop(['timestamp'],axis=1)
test=test.drop(['timestamp'],axis=1)
print("train shape:",train.shape)
print("test shape:",test.shape)

#userNo的最大值
userNo=max(train['user'].max(),test['user'].max())+1
print("userNo:",userNo)
#movieNo的最大值
itemNo=max(train['item'].max(),test['item'].max())+1
print("itemNo:",itemNo)

rating_train=pt.zeros((itemNo,userNo))
rating_test=pt.zeros((itemNo,userNo))
for index,row in train.iterrows():
    #train数据集进行遍历
    rating_train[int(row['item'])][int(row['user'])]=row['rating']
print(rating_train[0:3][1:10])
for index,row in test.iterrows():
    rating_test[int(row['item'])][int(row['user'])] = row['rating']

def normalizeRating(rating_train):
    m,n=rating_train.shape
    # 每部电影的平均得分
    rating_mean=pt.zeros((m,1))
    #所有电影的评分
    all_mean=0
    for i in range(m):
        #每部电影的评分
        idx=(rating_train[i,:]!=0)
        rating_mean[i]=pt.mean(rating_train[i,idx])
    tmp=rating_mean.numpy()
    tmp=np.nan_to_num(tmp)        #对值为NaN进行处理，改成数值0
    rating_mean=pt.tensor(tmp)
    no_zero_rating=np.nonzero(tmp)                #numpyy提取非0元素的位置
    # print("no_zero_rating:",no_zero_rating)
    no_zero_num=np.shape(no_zero_rating)[1]   #非零元素的个数
    print("no_zero_num:",no_zero_num)
    all_mean=pt.sum(rating_mean)/no_zero_num
    return rating_mean,all_mean

rating_mean,all_mean=normalizeRating(rating_train)
print("all mean:",all_mean)

#训练集分批处理
loader = Data.DataLoader(
    dataset=rating_train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # 最新批数据
    shuffle=False           # 是否随机打乱数据
)

loader2 = Data.DataLoader(
    dataset=rating_test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # 最新批数据
    shuffle=False           # 是否随机打乱数据
)

class MF(pt.nn.Module):
    def __init__(self,userNo,itemNo,num_feature=20):
        super(MF, self).__init__()
        self.num_feature=num_feature     #num of laten features
        self.userNo=userNo               #user num
        self.itemNo=itemNo               #item num
        self.bi=pt.nn.Parameter(pt.rand(self.itemNo,1))    #parameter
        self.bu=pt.nn.Parameter(pt.rand(self.userNo,1))    #parameter
        self.U=pt.nn.Parameter(pt.rand(self.num_feature,self.userNo))    #parameter
        self.V=pt.nn.Parameter(pt.rand(self.itemNo,self.num_feature))    #parameter

    def mf_layer(self,train_set=None):
        # predicts=all_mean+self.bi+self.bu.t()+pt.mm(self.V,self.U)
        predicts =self.bi + self.bu.t() + pt.mm(self.V, self.U)
        return predicts

    def forward(self, train_set):
        output=self.mf_layer(train_set)
        return output


num_feature=2    #k
mf=MF(userNo,itemNo,num_feature)
mf
print("parameters len:",len(list(mf.parameters())))
param_name=[]
params=[]
for name,param in mf.named_parameters():
    param_name.append(name)
    print(name)
    params.append(param)
# param_name的参数依次为bi,bu,U,V

lr=0.3
_lambda=0.001
loss_list=[]
optimizer=pt.optim.SGD(mf.parameters(),lr)
# 对数据集进行训练
for epoch in range(1000):
    optimizer.zero_grad()
    output=mf(train)
    loss_func=pt.nn.MSELoss()
    # loss=loss_func(output,rating_train)+_lambda*(pt.sum(pt.pow(params[2],2))+pt.sum(pt.pow(params[3],2)))
    loss = loss_func(output, rating_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss)

print("train loss:",loss)

#评价指标rmse
def rmse(pred_rate,real_rate):
    #使用均方根误差作为评价指标
    loss_func=pt.nn.MSELoss()
    mse_loss=loss_func(pred_rate,real_rate)
    rmse_loss=pt.sqrt(mse_loss)
    return rmse_loss

# 测试网络
#测试时测试的是原来评分矩阵为0的元素，通过模型将为0的元素预测一个评分，所以需要找寻评分矩阵中原来元素为0的位置。
prediction=output[np.where(rating_train==0)]
#评分矩阵中元素为0的位置对应测试集中的评分
rating_test=rating_test[np.where(rating_train==0)]
rmse_loss=rmse(prediction,rating_test)
print("test loss:",rmse_loss)

plt.clf()
plt.plot(range(epoch+1),loss_list,label='Training data')
# plt.plot(range(epoch+1),loss_list,label='Training data')
plt.title("The MovieLens Dataset Learning Curve")
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()