import sklearn
import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange,tnrange,tqdm_notebook
import random

from peld_src.utils import Emotion_dict

import tensorflow.compat.v1 as tf

from data_utils.beam_final import retore_LSTM, restore_Emo
from peld_src.tf_main import vad_to_emo
from peld_src.train import FocalLoss

tf.disable_v2_behavior()

from src.utils import pre_logger
from src.configuration import ChatConfig
from data_utils.prepare_dialogue_data import get_word_count, read_emotion_words, construct_vocab, construct_word_dict, \
    write_test_data_beam
from data_utils.prepare_dialogue_data import read_training_file, align_sentence_length, get_predict_train_response_data
from data_utils.prepare_dialogue_data import read_emotion_label, align_batch_size, shuffle_train_data, get_word_list
from data_utils.prepare_dialogue_data import read_word_embeddings, filter_test_sentence_length, write_test_data
from data_utils.prepare_dialogue_data import filter_sentence_length, read_stop_words, read_total_embeddings
from data_utils.prepare_dialogue_data import align_test_batch_size
from tensorflow import keras

from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.utils import np_utils

from src.model import EmotionChatMachine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_together(model, args, train_dataloader, valid_dataloader, test_dataloader,session):

    # 预热的总步数，是训练集数量的5%
    num_warmup_steps = int(0.05 * args.train_length)
    # 整个训练过程的总步数=训练集数量*epoch个数
    num_training_steps = len(train_dataloader) * args.epochs
    # 模型训练时的优化器，lr是学习率
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    # 学习率预热，在预热期间，从0慢慢增加到adamW中的学习率，在预热阶段之后创建一个schedule，使其学习率从优化器中的初始lr线性降低到0
    print("reach here 1")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)  # PyTorch scheduler
    print("reach here 2")
    train_logs = []
    valid_logs = []
    test_logs = []

    model.zero_grad()

    # EmoDS的部分
    chat_config = ChatConfig(args.config_file)

    print("Now prepare data!\n")
    print("Read stop words!\n")
    stop_words = read_stop_words(args.stop_words_file)
    # 构造通用词汇的词汇表
    print("Construct vocab first\n")
    # 得到词嵌入（float组成的向量），词嵌入中词的index，词嵌入中所有的词
    total_embeddings, total_word2id, total_word_list = read_total_embeddings(args.embedding_file, args.max_vocab_size)
    #[情感词]=word count
    pre_word_count = get_word_count(args.pre_train_word_count_file, chat_config.word_count)
    # 构造好的情感词典dic[情感类别代号][情感词]=情感词word_count?
    emotion_words_dict = read_emotion_words(args.emotion_words_dir, pre_word_count)
    # 处理情感词和通用词，得到全部词的list
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
                                chat_config.emotion_vocab_size, args.unk)
    # 全部词的词典，0是unk,1是start letter,2是end letter
    word_dict = construct_word_dict(word_list, args.unk, args.start_symbol, args.end_symbol)
    # 根据index来查Word
    id2words = {idx: word for word, idx in word_dict.items()}
    # 其实word_unk_id就是0
    word_unk_id = word_dict[args.unk]
    # word_start_id是1
    word_start_id = word_dict[args.start_symbol]
    # word_end_id是2
    word_end_id = word_dict[args.end_symbol]
    final_word_list = get_word_list(id2words)

    print("Read word embeddings!\n")
    # 读所有的词向量
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)
    # print("Read training data!\n")
    # # 读用于训练的--上文、回复、情感标签
    # train_post_data = read_training_file(args.post_file, word_dict, args.unk)
    # train_response_data = read_training_file(args.response_file, word_dict, args.unk)
    # # 待读取emotion label
    # emotion_chat_machine = EmotionChatMachine(args.config_file, session, word_dict, embeddings,
    #                                           chat_config.generic_word_size + 3, word_start_id, word_end_id,
    #                                           "emotion_chat_machine")
    checkpoint_path = os.path.join(args.checkpoint_dir, "dialogue-model")


    # start training
    for i in tnrange(0, args.epochs, desc='Epoch'):
        # torch.cuda.empty_cache()
        # print("<" + "=" * 22 + F" Epoch {_} " + "=" * 22 + ">")
        # Calculate total loss for this epoch
        batch_loss = 0
        train_accuracy, nb_train_steps = 0, 0

        # if i != 0 and i % 3 == 0:
            # session.run(emotion_chat_machine.lr_decay_op)
        for step, batch in enumerate(train_dataloader,0):
            torch.cuda.empty_cache()
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()
            pred_list = np.array([])
            labels_list = np.array([])

            # Add batch to GPU
            # 原来是..cuda(args.device)改成了.to(device)
            batch = tuple(t.to(device) for t in batch)
            # 取一个batch中需要和有用的数据
            b_index, b_input_ids_1, b_post_length, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch
            # print("post langth here")
            # print(b_post_length)
            # exit()
            # 将输入、个性、初始情绪都传入模型，得到FC层输出的概率，形状是[batch size,classes num]
            print(b_input_ids_1.device)
            print(b_input_ids_2.device)
            print("device check here")
            logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)
            # 均方误差，参数估计值与参数真值之差平方的期望值，也就是简单的预测值和真实值之间的差的平方
            if args.loss_function == 'MSE':
                loss_fct     = nn.MSELoss()
                loss         = loss_fct(logits, b_response_emo)
                # 返回的就是一个新的tensor，但是没有梯度了
                pred_emotion = logits.detach().to('cpu').numpy()
                # 模型得到的预测情感
                pred_flat    = vad_to_emo(pred_emotion, Emotion_dict).flatten()
                # 数据集中的真实情感
                labels_flat  = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()
            else:
                # 交叉熵损失函数
                if args.loss_function == 'CE': # cross entropy:
                    loss_fct = nn.CrossEntropyLoss()
                elif args.loss_function == 'Focal': # Focal loss:
                    loss_fct = FocalLoss()

                # 用选定的损失函数（2种）计算损失
                loss        = loss_fct(logits, b_labels)
                logits      = logits.detach().to('cpu').numpy()
                label_ids   = b_labels.to('cpu').numpy()
                # 从logits里选可能性最大的那个情感,flatten返回一个折叠成一维的数组
                pred_flat   = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()

            # 把这一次的batch得到的pred_list的情感放入list
            pred_list   = np.append(pred_list, pred_flat).tolist()
            labels_list = np.append(labels_list, labels_flat)
            df_metrics  = pd.DataFrame({'Epoch':args.epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})

            nb_train_steps += 1

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.cuda.empty_cache()

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update learning rate schedule
            scheduler.step()
            optimizer.zero_grad()

            # Update tracking variables
            batch_loss += loss.item()
            exit()
            # Process the data of Reply Generation
            this_train_res_data = b_input_ids_3.cpu().numpy().tolist()
            this_post_data = b_input_ids_2.cpu().numpy().tolist()
            this_post_len = b_post_length.cpu().numpy().tolist()
            this_emotion_labels = pred_list
            print("end here")
            print(this_emotion_labels)

            this_train_res_data,this_predict_res_data = get_predict_train_response_data(this_train_res_data,word_start_id,word_end_id,word_unk_id,chat_config.max_len)
            # print(step)
            this_emotion_mask = emotion_chat_machine.get_emotion_word_masks(pred_list)
            # this_post_data, this_post_len, this_train_res_data, this_predict_res_data, this_emotion_labels,this_emotion_mask \
                # = emotion_chat_machine.get_batch(b_input_ids_2,b_post_length,b_input_ids_3,b_input_ids_3_pre,pred_list,step)
            # print(b_input_ids_2)
            # print(b_post_length)
            # print(b_input_ids_3_pre)
            # print(pred_list)
            loss = emotion_chat_machine.train_step(this_post_data, this_post_len, this_train_res_data,
                                                   this_predict_res_data, this_emotion_labels, this_emotion_mask)

            entropy_loss, reg_loss, total_loss = loss
            torch.cuda.empty_cache()
            # print("Epoch=%d, batch=%d, total loss=%f, entropy loss=%f, reg_loss=%f\n" %
            #       ((i + 1), (j + 1), total_loss, entropy_loss, reg_loss))

        print("Saving parameters\n")
        emotion_chat_machine.saver.save(emotion_chat_machine.session, checkpoint_path,
                                        global_step=(i * args.train_length))

