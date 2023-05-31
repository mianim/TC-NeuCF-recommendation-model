import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors,layers, dropout):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        # self.num_factors_mf = num_factors
        # self.dropout = dropout

        self.embed_user_mf = nn.Embedding(num_embeddings=self.num_users,
                                          embedding_dim=self.num_factors_mf)
        self.embed_item_mf = nn.Embedding(num_embeddings=self.num_items,
                                          embedding_dim=self.num_factors_mf)


        self.affine_output = nn.Linear(in_features=self.num_factors_mf,
                                       out_features=1)

        self.logistic = nn.Sigmoid()
        self._init_wieght_()

    def _init_wieght_(self):
        # 将权重初始化成高斯分布 userEmbedding有一个weight
        # itemEmbedding有一个weight 所以要初始化两个weight
        nn.init.normal_(self.embed_user_mf.weight, std=0.01)
        nn.init.normal_(self.embed_item_mf.weight, std=0.01)


        nn.init.xavier_uniform_(self.affine_output.weight)

        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, user, item):
        '''
        将用户 和 物品输入到对应的embedding 层
        然后将两个embedding 进行内积
        将内积的结果输入到预测层（线性层进行预测）
        最后输出
        '''
        embed_user_GMF = self.embed_user_mf(user)
        embed_item_GMF = self.embed_item_mf(item)

        mf_vector = torch.mul(embed_user_GMF, embed_item_GMF)
        vector = torch.cat([mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()


class MLP(nn.Module):
    def __init__(self, num_users, num_items, num_factors,layers, dropout):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors_mf = num_factors
        self.num_factors_mlp = int(layers[0] / 2)
        self.layers = layers  # Layers of MLP
        self.dropout = dropout

        self.embed_user_mlp = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.num_factors_mlp)
        self.embed_item_mlp = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.num_factors_mlp)

        # 全连接层
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(layers, layers[1:]):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=layers[-1],
                                       out_features=1)

        self.logistic = nn.Sigmoid()
        self._init_wieght_()

    def _init_wieght_(self):
        # 将权重初始化成高斯分布 userEmbedding有一个weight
        # itemEmbedding有一个weight 所以要初始化两个weight
        nn.init.normal_(self.embed_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embed_item_mlp.weight, std=0.01)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


        nn.init.xavier_uniform_(self.affine_output.weight)

        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        '''
        将用户 和 物品输入到对应的embedding 层
        然后将两个embedding 进行内积
        将内积的结果输入到预测层（线性层进行预测）
        最后输出
        '''
        user_embedding_mlp = self.embed_user_mlp(user_indices)
        item_embedding_mlp = self.embed_item_mlp(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp],
                               dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()



class NeuCF(nn.Module):
    """

    Parameters
    ----------
    num_users : int
        Number of unique users.
    num_items : int
        Number of unique items.
    num_factors : int
        Embedding size.
    layers : list
        Layers of MLP.
    dropout : float
        Dropout rate.

    Returns
    -------
    None.

    """
    def __init__(self, num_users, num_items, num_factors, layers, dropout):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors_mf = num_factors
        self.num_factors_mlp = int(layers[0]/2)
        self.layers = layers # Layers of MLP
        self.dropout = dropout

        # MLP
        self.embed_user_mlp = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.num_factors_mlp)
        self.embed_item_mlp = nn.Embedding(num_embeddings=self.num_items,
                                         embedding_dim=self.num_factors_mlp)

        # MF
        self.embed_user_mf = nn.Embedding(num_embeddings=self.num_users,
                                        embedding_dim=self.num_factors_mf)
        self.embed_item_mf = nn.Embedding(num_embeddings=self.num_items,
                                        embedding_dim=self.num_factors_mf)

        # 全连接层
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(layers, layers[1:]):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())


        # 合并所有信息
        self.affine_output = nn.Linear(in_features=layers[-1] +
                                       self.num_factors_mf,
                                       out_features=1)

        # 激活函数
        self.logistic = nn.Sigmoid()
        self.init_weight()


    def init_weight(self):
        nn.init.normal_(self.embed_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embed_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embed_user_mf.weight, std=0.01)
        nn.init.normal_(self.embed_item_mf.weight, std=0.01)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                
        nn.init.xavier_uniform_(self.affine_output.weight)

        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.data.zero_()


    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embed_user_mlp(user_indices)
        item_embedding_mlp = self.embed_item_mlp(item_indices)

        user_embedding_mf = self.embed_user_mf(user_indices)
        item_embedding_mf = self.embed_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp],
                               dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()


def recommend(item_list, movies_df, model, test_loader, top_k, user_id, device):
    """

    Parameters
    ----------
    item_list : list
        List of unique items.
    movies_df : Pandas DataFrame
        DataFrame object that contains movie ids and titles.
    model : Torch model
        Model to be evaluated.
    test_loader : torch.utils.data.DataLoader
        Torch DataLoader object.
    top_k : int
        Top k instances to calculate metrics.
    user_id : int
        DESCRIPTION.
    device : torch.device
        Torch device object.

    Returns
    -------
    None.

    """

    for user, item, _,_,_ in test_loader:
        if user[0].item() == user_id:
            user = user.to(device)
            item = item.to(device)
    
            predictions = model(user, item)
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(item, indices).cpu().numpy().tolist()
            break


    return recommends
    # with open('recommendations.txt', 'w') as f:
    #     f.write("Recommendations for User: {} \n".format(user_id))
    #     print("\n\n 推荐的电影列表——用户: {} \n".format(user_id))
    #     for new_item_id in recommends:
    #         old_item_id = item_list[new_item_id]
    #         item_title = movies_df["title"][movies_df["item_id"]==old_item_id] \
    #                      .values[0]
    #         f.write("{}-) {} \n".
    #               format(recommends.index(new_item_id) + 1, item_title))
    #
    #         print("{}-) {}".
    #               format(recommends.index(new_item_id) + 1, item_title))


def precision(item_list, movies_df, model, test_loader, top_k,  device):
    hit=0
    n_recall = 0
    n_precision = 0
    user_movies=dict()

    for user, item, _ in test_loader:
        user_movies[user[0].item()]=item.tolist()

    for user in user_movies:
        rank=recommend(item_list, movies_df, model, test_loader, top_k, user, device)
        hit+=len(set(rank) & set(user_movies[user]))
        n_recall+=len(user_movies[user])
        n_precision+=top_k

    print("召回率：",hit / (1.0 * n_recall),"准确率：",hit / (1.0 * n_precision))


        # tu=test_loader[user]
        # rank=recommend(item_list, movies_df, model, test_loader, top_k, user_id, device)
        # print(rank)