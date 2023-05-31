import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.sparse as sp

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

import config
import data_utils
import model
import evaluate


def main():
    # Change directory to the parent dir
    os.chdir(os.path.dirname(os.getcwd()))
    
    # Device and sumamry writer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    
    # 加载数据
    rating = pd.read_csv(config.rating_data_path, 
                         sep="::", 
                         names = ['user_id', 'item_id', 'rating', 'timestamp'], 
                         engine='python')
    
    movies = pd.read_csv(config.movies_data_path, 
                         sep="::", 
                         names = ["item_id", "title", "genres"], 
                         engine='python',
                         encoding="ISO-8859-1")[["item_id", "title"]]

    # 新增时间属性
    rating["timestamp"] = rating["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
    # 追加
    # 白天时段
    rating["daytime"] = rating["timestamp"].apply(lambda x: 1 if 6 < int(x.strftime("%H")) < 20 else 0)
    # 周末
    rating["weekend"] = rating["timestamp"].apply(lambda x: 1 if x.weekday() in [5, 6] else 0)


    # 去除完全重复的行
    item_list = list(rating['item_id'].drop_duplicates())

    # 返回所有数据的唯一值
    num_users = rating['user_id'].nunique()
    num_items = rating['item_id'].nunique()

    # 分成训练集和测试集
    data = data_utils.NcfData(rating, config.num_neg, config.num_neg_test,
                              config.batch_size, config.seed)
    train_loader = data.get_train_instance()
    test_loader = data.get_test_instance()

    # 创建模型，设置损失 set loss and optimizer
    neucf_model = model.NeuCF(num_users, num_items, config.num_factors,
                              config.layers, config.dropout)
    # GMF 追加
    # neucf_model = model.GMF(num_users, num_items, config.num_factors,
    #                           config.layers, config.dropout)
    # MLP 追加
    # neucf_model = model.MLP(num_users, num_items, config.num_factors,
    #                           config.layers, config.dropout)

    neucf_model = neucf_model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(neucf_model.parameters(), lr=config.lr)

    # 训练模型————————————————————————————————————————————————————————————————————————————
    # Train the model
    # best_hr = 0
    # for epoch in range(1, config.epochs+1):
    #     neucf_model.train()
    #     start_time = time.time()
    #
    #     for user, item, label in train_loader:
    #         user = user.to(device)
    #         item = item.to(device)
    #         label = label.to(device)
    #
    #         optimizer.zero_grad()
    #         prediction = neucf_model(user, item)
    #         # loss = loss_function(prediction, label)
    #         # 追加
    #         loss = loss_function(prediction, label)
    #         loss.backward()
    #         optimizer.step()
    #         writer.add_scalar('Perfomance/Train_loss', loss.item(), epoch)
    #
    #     # Test the model
    #     neucf_model.eval()
    #     hr, ndcg = evaluate.metrics(neucf_model, test_loader,
    #                                 config.top_k, device)
    #
    #     # Add performance metrics to tensorboard
    #     writer.add_scalar("Perfomance/HR_10", hr, epoch)
    #     writer.add_scalar("Perfomance/NDCG_10", ndcg, epoch)
    #
    #     # Calculate and print the elapsed time
    #     elapsed_time = time.time() - start_time
    #     print("The time elapse of epoch {}".format(epoch) + " is: " +
    #             time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    #     print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(hr), np.mean(ndcg)))
    #
    #     # Save the best model in terms of hit ratio
    #     if hr > best_hr:
    #         best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
    #         if config.out:
    #             torch.save(neucf_model, "neucf_model.pth")
    #
    # # Add performance metrics to tensorboard
    # writer.add_hparams({"epoch": config.epochs,
    #                     "lr": config.lr,
    #                     "batch_size": config.batch_size,
    #                     "dropout": config.dropout},
    #                    {"HR_10": hr,
    #                     "NDCG_10": ndcg})
    #
    # # Close the tensorboard object and print best epoch
    # writer.close()
    # print("End. Best epoch {}: HR = {:.3f}, NDCG = {:.3f}".format(
    #                                         best_epoch, best_hr, best_ndcg))

    # 训练模型————————————————————————————————————————————————————————————————————————————
    # 新增时间上下文
    # Train the model
    best_hr = 0
    for epoch in range(1, config.epochs + 1):
        neucf_model.train()
        start_time = time.time()

        for user, item, label,daytime,weekend in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            daytime = daytime.to(device)
            weekend=weekend.to(device)


            optimizer.zero_grad()
            prediction = neucf_model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Perfomance/Train_loss', loss.item(), epoch)

        # Test the model
        neucf_model.eval()
        hr, ndcg = evaluate.metrics(neucf_model, test_loader,
                                    config.top_k, device)

        # Add performance metrics to tensorboard
        writer.add_scalar("Perfomance/HR_10", hr, epoch)
        writer.add_scalar("Perfomance/NDCG_10", ndcg, epoch)

        # Calculate and print the elapsed time
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(hr), np.mean(ndcg)))

        # Save the best model in terms of hit ratio
        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
            if config.out:
                torch.save(neucf_model, "neucf_model.pth")

    # Add performance metrics to tensorboard
    writer.add_hparams({"epoch": config.epochs,
                        "lr": config.lr,
                        "batch_size": config.batch_size,
                        "dropout": config.dropout},
                       {"HR_10": hr,
                        "NDCG_10": ndcg})

    # Close the tensorboard object and print best epoch
    writer.close()
    print("End. Best epoch {}: HR = {:.3f}, NDCG = {:.3f}".format(
        best_epoch, best_hr, best_ndcg))
    # 新增时间上下文

    # 实现推荐——————————————————————————————————————————————————————————————
    # recommends=model.recommend(item_list,
    #                 movies,
    #                 neucf_model,
    #                 test_loader,
    #                 config.recommend_n,
    #                 config.recommend_user,
    #                 device)
    # print("为ID为",config.recommend_user,"的用户推荐电影如下：")
    # for new_item_id in recommends:
    #     old_item_id=item_list[new_item_id]
    #     item_title = movies["title"][movies["item_id"] == old_item_id] \
    #         .values[0]
    #     print("{}- {}".
    #           format(recommends.index(new_item_id) + 1, item_title))
    # 实现推荐——————————————————————————————————————————————————————————————

    # 评测推荐系统准确度
    # print(test_loader)
    # model.precision(item_list,
    #                 movies,
    #                 neucf_model,
    #                 test_loader,
    #                 config.recommend_n,
    #                 device)


    
if __name__ == '__main__':
    main()
