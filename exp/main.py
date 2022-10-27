# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
import model


import torch
print(torch.cuda.is_available())
# CONFIG
# 配置项目

# 数据路径
DATA_PATH = '../Dyadic_PELD_CN_index.csv'
TEST_DATA_PATH = '../Dyadic_all.tsv'

# Step1：创建 ArgumentParser() 对象，就是一个解析器，这个解析器包含将命令行解析成 Python 数据类型所需的全部信息。
parser = argparse.ArgumentParser(description='')
# Step2：调用 add_argument() 方法添加参数
# Step3：使用 parse_args() 解析添加的参数
args = parser.parse_args()

# 然后设置args实例的一些参数
args.mode = 3
args.Senti_or_Emo = 'Emotion'
args.loss_function = 'CE' # CE or MSE or Focal
args.device = 0
args.SEED = 42
args.MAX_LEN = 30
args.batch_size = 48
args.lr = 1e-5
args.adam_epsilon = 1e-8
args.epochs = 800
args.result_name = args.Senti_or_Emo+'_Mode_'+str(args.mode)+'_'+args.loss_function+'_Epochs_'+str(args.epochs)+'_CN_index'+'.csv'

# LOAD DATA
from dataload import load_data,load_test_data
# 从data_path里加载训练数据、验证集数据和测试数据
train_length, train_dataloader, valid_dataloader, test_dataloader = load_data(args, DATA_PATH)
# test_dataloader = load_test_data(args,TEST_DATA_PATH)
args.train_length = train_length
## TRAIN THE MODEL

# 是model.py里面的Emo_Generation类
from model import Emo_Generation
from transformers import RobertaConfig, RobertaModel, PreTrainedModel, BertTokenizer
# 是tain.py里面的train_model函数
from train import train_model,test_model

# 原来是：.cuda(args.device)改成了.to(device)
# 加载预训练模型
torch.cuda.empty_cache()
model = Emo_Generation.from_pretrained('roberta-base', mode=args.mode).cuda(args.device)
train_model(model, args, train_dataloader, valid_dataloader, test_dataloader)
# test_model(test_dataloader,args)






