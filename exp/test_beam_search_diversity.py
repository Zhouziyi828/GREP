import os
import argparse
import random
import sys

import tensorflow.compat.v1 as tf
from tensorflow import keras

from data_utils.beam_final import retore_LSTM_random

tf.disable_v2_behavior()

from tensorflow.python.tools import inspect_checkpoint as chkp
from src.utils import pre_logger
from src.configuration import ChatConfig
from data_utils.prepare_dialogue_data import get_word_count, read_emotion_words, construct_vocab, construct_word_dict, \
    write_test_data_beam
from data_utils.prepare_dialogue_data import read_training_file, align_sentence_length, get_predict_train_response_data
from data_utils.prepare_dialogue_data import read_emotion_label, align_batch_size, shuffle_train_data, get_word_list
from data_utils.prepare_dialogue_data import read_word_embeddings, filter_test_sentence_length, write_test_data
from data_utils.prepare_dialogue_data import filter_sentence_length, read_stop_words, read_total_embeddings
from data_utils.prepare_dialogue_data import align_test_batch_size
from data_utils.beam import select_best_response, index_test_data, read_beam_test
from src.model import EmotionChatMachine
import numpy as np
from data_utils.lstm_classifier import LstmClassifier, add_emo_beam, response_to_indexs_b


def generate_response(config_file, pre_train_word_count_file, emotion_words_dir, embedding_file,checkpoint_dir,checkpoint_dir_lstm, max_vocab_size, test_post_file, test_label_file):
    """
    generate response from checkpoint
    :param config_file:
    :param pre_train_word_count_file:
    :param emotion_words_dir:
    :param embedding_file:
    :param session:
    :param checkpoint_dir:
    :param max_vocab_size:
    :param test_post_file:
    :param test_label_file:
    :param log_name:
    :param restore_model:
    :return:
    """
    #logger = pre_logger(log_name)

    chat_config = ChatConfig(config_file)

    print("Now prepare data!\n")
    print("Read stop words!\n")
    stop_words = read_stop_words(FLAGS.stop_words_file)

    print("Construct vocab first\n")
    total_embeddings, total_word2id, total_word_list = read_total_embeddings(embedding_file, max_vocab_size)
    pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
                                chat_config.emotion_vocab_size, FLAGS.unk)
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    id2words = {idx: word for word, idx in word_dict.items()}
    word_unk_id = word_dict[FLAGS.unk]
    word_start_id = word_dict[FLAGS.start_symbol]
    word_end_id = word_dict[FLAGS.end_symbol]
    final_word_list = get_word_list(id2words)

    print("Read word embeddings!\n")
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)

    print("Read test data\n")
    test_post_data = read_training_file(test_post_file, word_dict, FLAGS.unk)
    test_label_data = read_emotion_label(test_label_file)

    # 把不符合长度的除掉
    print("filter test post data length!\n")
    test_post_data, test_label_data = filter_test_sentence_length(test_post_data, test_label_data, chat_config.min_len,
                                                                  chat_config.max_len)

    print("Number of length <= 10 sentences: %d\n" % len(test_post_data))
    test_post_data_length = [len(post_data) for post_data in test_post_data]
    test_length = len(test_post_data)
    # 补齐句子
    print("Align sentence length by padding!\n")
    test_post_data = align_sentence_length(test_post_data, chat_config.max_len, word_unk_id)
    test_post_data, test_post_data_length, test_label_data = \
        align_test_batch_size(test_post_data, test_post_data_length, test_label_data, chat_config.batch_size)


    print("Define model\n")

        # emotion_chat_machine = EmotionChatMachine(config_file, session, word_dict, embeddings,
        #                                       chat_config.generic_word_size + 3, word_start_id, word_end_id,
        #                                       "emotion_chat_machine")
    checkpoint_path = os.path.join(checkpoint_dir, "dialogue-model-1370543")
    checkpoint_path_l = os.path.join(checkpoint_dir_lstm, "check_path_lstm-672588")


    print("Generate test data!\n")
    test_batch = int(len(test_post_data) / chat_config.batch_size)
    generate_words, scores,this_post_data,this_emotion_labels = restore_Emo(checkpoint_path, config_file, word_dict, embeddings, chat_config, word_start_id,word_end_id,test_post_data,test_post_data_length,test_label_data)

    generate_data = retore_LSTM_r(chat_config,checkpoint_path_l,generate_words,scores,this_post_data,this_emotion_labels,8, word_unk_id,embeddings,word_dict,id2words)
    print("test length first check ")
    print(len(generate_data))
    # graph = tf.Graph()
    # with tf.Session(graph=graph) as sess:
    #     emotion_chat_machine = EmotionChatMachine(config_file, sess, word_dict, embeddings,
    #                                               chat_config.generic_word_size + 3, word_start_id, word_end_id,
    #                                               "emotion_chat_machine")
    #     emotion_chat_machine.saver.restore(sess, checkpoint_path)
    # graph_other = tf.Graph()
    # with tf.Session(graph=graph_other) as sess_other:
    #     lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8, 15, True, sess_other)
    #     lstm_emotion_machine.saver.restore(sess_other, checkpoint_path)
    '''
    for k in range(test_batch):
        this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
            emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
        generate_words, scores, new_embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,this_emotion_labels,this_emotion_mask)
        print("size here:")
        print(generate_words.shape)
        generate words: [batch, beam, max len]  &&  scores: [batch, beam]
        best_generate_sen = select_best_response(generate_words, scores, this_post_data, this_emotion_labels,emotion_words_dict, chat_config.batch_size, stop_words, word_start_id ,word_end_id, word_unk_id,lstm_classi,emotion_chat_machine)
        generate_data.extend(best_generate_sen)
    '''
    print("test_length check check")
    print(test_length)
    generate_data = generate_data[: test_length]

    test_label_data = test_label_data[: test_length]

    write_test_data_beam(test_post_data,generate_data, FLAGS.generate_response_file, id2words, test_label_data)

def restore_Emo(checkpoint,config_file,word_dict,embeddings,chat_config,word_start_id,word_end_id,test_post_data,test_post_data_length,test_label_data):
    graph=tf.Graph()
    with tf.Session(graph=graph) as sess:
        emotion_chat_machine = EmotionChatMachine(config_file, sess, word_dict, embeddings,
                                              chat_config.generic_word_size + 3, word_start_id, word_end_id,
                                              "emotion_chat_machine")
        emotion_chat_machine.saver.restore(sess,checkpoint)
        test_batch = int(len(test_post_data) / chat_config.batch_size)
        print("test_batch")
        print(test_batch)
        generate_data = []
        scores_data=[]
        this_post_data_l,this_emotion_labels_l=[],[]
        for k in range(test_batch):
            this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
                emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
            generate_words, scores,embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,
                                                                                        this_emotion_labels,
                                                                                        this_emotion_mask)
            generate_data.append(generate_words)
            scores_data.append(scores)
            this_post_data_l.append(this_post_data)
            this_emotion_labels_l.append(this_emotion_labels)

        return generate_data,scores_data,this_post_data_l,this_emotion_labels_l

def retore_LSTM(chat_config,checkpoint_path,generate_words,scores,this_post_data,this_emotion_labels,batch_size, word_unk_id,embeddings,word_dict,id2words):
    print("length of embedding")
    print(len(embeddings))
    graph_other=tf.Graph()
    with tf.Session(graph=graph_other) as sess_other:
        lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8,15, True, sess_other)
        lstm_emotion_machine.saver.restore(sess_other, checkpoint_path)
        best_responses=[]
        # print("check again")
        # print(len(generate_words))
        # print(len(scores))
        # print(len(this_post_data))
        # print(len(this_emotion_labels))


        for generate_words_b,scores_b,this_post_data_b,this_emotion_labels_b in zip(generate_words,scores,this_post_data,this_emotion_labels):
            generate_words_b = index_test_data(generate_words_b, id2words)


            generate_words_b = change_file_format(generate_words_b)


            this_emotion_labels_b = add_emo_beam(chat_config.beam_size, this_emotion_labels_b)

            print("this_emotion_labels_b l")
            print(len(this_emotion_labels_b))
            test_responses_b, test_lens_b, this_emotion_labels_b = read_beam_test(generate_words_b, this_emotion_labels_b,
                                                                            chat_config.max_len)

            # for b in test_responses_b:
            #     print("in test_responses_b length")
            #     print(len(b))
            print("test_responses_b")
            print(len(test_responses_b))
            test_responses_b = response_to_indexs_b(test_responses_b, word_dict, word_unk_id, chat_config.max_len)
            # for b in test_responses_b:
            #     print("in test_responses_b length")
            #     print(len(b))
            this_labels_scores = lstm_emotion_machine.beam_predict_step(test_responses_b, test_lens_b, this_emotion_labels_b)

            scores_mul_b = []
            max_col_b = []
            total_labels_b = []
            for ge_score, emo_score in zip(scores_b, this_labels_scores):
                # emo_score_b_max=np.max(emo_score,axis=1)
                scores_mul_i=[]


                for ge,emo in zip(ge_score,emo_score):
                    scores_mul_bl=ge*emo
                    scores_mul_i.append(scores_mul_bl)
                scores_mul_b.append(scores_mul_i)
            print("scores chekc here")
            print(scores_mul_b)
            # scores_mul_b[batch,beam_size]
            for scores_ in scores_mul_b:
                max_index = scores_.index(max(scores_))
                max_col_b.append(max_index)
            print("max check here")
            print(max_col_b)
            # max_col:[batch_size]
            for i in range(batch_size):
                total_labels_b.append(generate_words_b[i][max_col_b[i]])
                # total_labels_b:[8]
            best_responses.extend(total_labels_b)
            print("Check!!!")
            print(best_responses)
            # best:[67*8]
        return best_responses

def retore_LSTM_r(chat_config,checkpoint_path,generate_words,scores,this_post_data,this_emotion_labels,batch_size, word_unk_id,embeddings,word_dict,id2words):
    print("length of embedding")
    print(len(embeddings))
    graph_other = tf.Graph()
    with tf.Session(graph=graph_other) as sess_other:
        lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8, 15, True, sess_other)
        lstm_emotion_machine.saver.restore(sess_other, checkpoint_path)
        random_responses = []
        print("check again")
        print(len(scores))

        for generate_words_b, scores_b, this_post_data_b, this_emotion_labels_b in zip(generate_words, scores,
                                                                                       this_post_data,
                                                                                       this_emotion_labels):
            generate_words_b = index_test_data(generate_words_b, id2words)

            generate_words_b = change_file_format(generate_words_b)

            this_emotion_labels_b = add_emo_beam(chat_config.beam_size, this_emotion_labels_b)

            print("this_emotion_labels_b l")
            print(len(this_emotion_labels_b))
            test_responses_b, test_lens_b, this_emotion_labels_b = read_beam_test(generate_words_b,
                                                                                  this_emotion_labels_b,
                                                                                  chat_config.max_len)
            for b in test_responses_b:
                print("in test_responses_b length")
                print(len(b))
            print("test_responses_b")
            print(len(test_responses_b))
            test_responses_b = response_to_indexs_b(test_responses_b, word_dict, word_unk_id, chat_config.max_len)
            # for b in test_responses_b:
            #     print("in test_responses_b length")
            #     print(len(b))
            this_labels_scores = lstm_emotion_machine.beam_predict_step(test_responses_b, test_lens_b,
                                                                        this_emotion_labels_b)

            scores_mul_b = []
            random_col_b = []
            total_labels_b = []
            for ge_score, emo_score in zip(scores_b, this_labels_scores):
                # emo_score_b_max=np.max(emo_score,axis=1)
                scores_mul_i = []
                for ge, emo in zip(ge_score, emo_score):
                    scores_mul_bl = ge * emo
                    scores_mul_i.append(scores_mul_bl)
                scores_mul_b.append(scores_mul_i)
            # scores_mul_b[batch,beam_size]
            for scores_ in scores_mul_b:
                random_index = random.randint(0, len(scores_) - 1)
                random_col_b.append(random_index)
            for i in range(batch_size):
                total_labels_b.append(generate_words_b[i][random_col_b[i]])
            random_responses.extend(total_labels_b)
        return random_responses



def read_test_sen(x_test, word_dict, unk="<unk>"):
    unk_id = word_dict[unk]
    train_data = []
    word_indexes = [word_dict[word] if word in word_dict else unk_id for word in x_test]
    train_data.append(word_indexes)
    return train_data

def change_file_format(origin):
    # [origin]的形状是：[batch_size,beam_size,max_len]
    start_symbol = "<ss>"
    end_symbol = "<es>"
    new=[]

    # f1 = open(file1, "r", encoding="utf-8")
    # f2 = open(file2, "w", encoding="utf-8")
    for line in origin:
        new_b=[]
        for line_b in line:
            # line_str=''.join(line_b)
            words = line_b.strip().split()
            words = words[1:]
            if start_symbol in words:
                start_index = words.index(start_symbol) + 1
            else:
                start_index = 0
            if end_symbol in words:
                end_index = words.index(end_symbol)
            else:
                end_index = len(words)
            selected_words = words[start_index: end_index]
            sentence = " ".join('%s' %a for a in selected_words)
            new_b.append(sentence)

        new.append(new_b)
    return new



def test(session,config_file, pre_train_word_count_file, emotion_words_dir, post_file, response_file, emotion_label_file,
          embedding_file, train_word_count, checkpoint_dir, max_vocab_size, test_post_file, test_label_file,checkpoint_dir_lstm):

    chat_config = ChatConfig(config_file)
    # f = open('log.log.emo', 'w')
    # sys.stdout = f
    # #
    # sys.stderr = f

    print("Now prepare data!\n")
    print("Read stop words!\n")
    stop_words = read_stop_words(FLAGS.stop_words_file)

    print("Construct vocab first\n")
    total_embeddings, total_word2id, total_word_list = read_total_embeddings(embedding_file, max_vocab_size)
    pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
                                chat_config.emotion_vocab_size, FLAGS.unk)
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    id2words = {idx: word for word, idx in word_dict.items()}
    word_unk_id = word_dict[FLAGS.unk]
    word_start_id = word_dict[FLAGS.start_symbol]
    word_end_id = word_dict[FLAGS.end_symbol]
    final_word_list = get_word_list(id2words)

    print("Read word embeddings!\n")
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)

    # emotion_chat_machine = EmotionChatMachine(config_file, session, word_dict, embeddings,
    #                                           chat_config.generic_word_size + 3, word_start_id, word_end_id,
    #                                           "emotion_chat_machine")
    checkpoint_path_ecm = os.path.join(checkpoint_dir, "dialogue-model-913543")
    # chkp.print_tensors_in_checkpoint_file(checkpoint_path, tensor_name=None, all_tensors=True)
    emotion_chat_machine=keras.models.load_model(checkpoint_dir)
    lstm_classi=keras.models.load_model(checkpoint_dir_lstm)

    while True:
        user_text = input('Input Chat Sentence:')
        if user_text in ('exit', 'quit'):
            exit(0)
        user_lable=input('Input Chat Emotion:')
        x_test = list(user_text.lower())
        x_label=[int(user_lable.lower())]
        x_length=[len(x) for x in x_test]
        x_test = read_test_sen(x_test, word_dict, FLAGS.unk)
        print("x_ test here")
        print(x_test)
        x_test = align_sentence_length(x_test, chat_config.max_len, word_unk_id)
        test_post_data, test_post_data_length, test_label_data = \
            align_test_batch_size(x_test, x_length, x_label, chat_config.batch_size)
        test_batch = int(len(test_post_data) / chat_config.batch_size)
        generate_data = []
        for k in range(test_batch):
            this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
            emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
            print("test post data shape here")
            print(len(test_post_data))
            print(len(test_post_data[0]))
            generate_words, new_embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,this_emotion_labels, this_emotion_mask)
            generate_data.extend(generate_words)
        generate_data = generate_data[: 1]
        for res in generate_data:
            words = [id2words[index] for index in res]
            words=change_file_format(words)
            print(words)

    # g1 = tf.Graph()
    # sess = tf.Session(config=tf.ConfigProto(
    #     allow_soft_placement=True, log_device_placement=True),graph=g1)
    # embeddings, word_dict, id2word = read_total_embeddings(embedding_file, 40000)
    # # tf.reset_default_graph()
    # with g1.as_default():
    #     lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8,
    #                                           15, True, sess, learning_rate=0.1)
    #     checkpoint_path = os.path.join(os.path.join(model_path, "data_utils/check_path_lstm"), "check_path_lstm-11728")
    #
    #     lstm_emotion_machine.saver.restore(lstm_emotion_machine.sess, checkpoint_path)
    # return lstm_emotion_machine,sess,g1
def final_test(session,config_file,pre_train_word_count_file, emotion_words_dir,embedding_file,checkpoint_dir, max_vocab_size,x_test,x_label,FLAGS):
    chat_config = ChatConfig(config_file)

    print("Construct vocab first\n")
    total_embeddings, total_word2id, total_word_list = read_total_embeddings(embedding_file, max_vocab_size)
    pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
                                chat_config.emotion_vocab_size, FLAGS.unk)
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    id2words = {idx: word for word, idx in word_dict.items()}
    word_unk_id = word_dict[FLAGS.unk]
    word_start_id = word_dict[FLAGS.start_symbol]
    word_end_id = word_dict[FLAGS.end_symbol]
    final_word_list = get_word_list(id2words)

    print("Read word embeddings!\n")
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)

    # emotion_chat_machine = EmotionChatMachine(config_file, session, word_dict, embeddings,
    #                                           chat_config.generic_word_size + 3, word_start_id, word_end_id,
    #                                           "emotion_chat_machine")
    checkpoint_path = os.path.join(checkpoint_dir, "dialogue-model-1361319")

    # chkp.print_tensors_in_checkpoint_file(checkpoint_path, tensor_name=None, all_tensors=True)
    emotion_chat_machine=keras.models.load_model(checkpoint_path)
    lstm_cla=keras.models.load_model()
    x_length = [len(x) for x in x_test]
    x_test = read_test_sen(x_test, word_dict, FLAGS.unk)
    print("x_ test here")
    print(x_test)
    x_test = align_sentence_length(x_test, chat_config.max_len, word_unk_id)
    test_post_data, test_post_data_length, test_label_data = \
        align_test_batch_size(x_test, x_length, x_label, chat_config.batch_size)
    test_batch = int(len(test_post_data) / chat_config.batch_size)
    generate_data = []
    for k in range(test_batch):
        this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
            emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
        print("test post data shape here")
        print(len(test_post_data))
        print(len(test_post_data[0]))
        generate_words, new_embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,
                                                                            this_emotion_labels, this_emotion_mask)
        generate_data.extend(generate_words)
    generate_data = generate_data[: 1]
    for res in generate_data:
        words = [id2words[index] for index in res]
        words = change_file_format(words)
        print(words)

def train(config_file, pre_train_word_count_file, emotion_words_dir, post_file, response_file, emotion_label_file,
          embedding_file, train_word_count, session, checkpoint_dir, max_vocab_size, test_post_file, test_label_file,
          log_name):
    # 训练的函数
    """
    train the dialogue model
    :param config_file:
    :param pre_train_word_count_file:
    :param emotion_words_dir:
    :param post_file:
    :param response_file:
    :param emotion_label_file:
    :param embedding_file:
    :param train_word_count:
    :param session:
    :param checkpoint_dir:
    :param max_vocab_size:
    :param test_post_file:
    :param test_label_file:
    :param log_name: log file name
    :return:
    """
    # logger = pre_logger(log_name)

    #
    chat_config = ChatConfig(config_file)

    print("Now prepare data!\n")
    print("Read stop words!\n")
    stop_words = read_stop_words(FLAGS.stop_words_file)
    # 构造通用词汇的词汇表
    print("Construct vocab first\n")
    # 得到词嵌入（float组成的向量），词嵌入中词的index，词嵌入中所有的词
    total_embeddings, total_word2id, total_word_list = read_total_embeddings(embedding_file, max_vocab_size)
    # [情感词]=word count
    pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    # 构造好的情感词典dic[情感类别代号][情感词]=情感词word_count?
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    # 处理情感词和通用词，得到全部词的list
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
                                chat_config.emotion_vocab_size, FLAGS.unk)
    # 全部词的词典，0是unk,1是start letter,2是end letter
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    # 根据index来查Word
    id2words = {idx: word for word, idx in word_dict.items()}
    # 其实word_unk_id就是0
    word_unk_id = word_dict[FLAGS.unk]
    # word_start_id是1
    word_start_id = word_dict[FLAGS.start_symbol]
    # word_end_id是2
    word_end_id = word_dict[FLAGS.end_symbol]
    final_word_list = get_word_list(id2words)

    print("Read word embeddings!\n")
    # 读所有的词向量
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)

    print("Read training data!\n")
    # 读用于训练的--上文、回复、情感标签
    train_post_data = read_training_file(post_file, word_dict, FLAGS.unk)
    train_response_data = read_training_file(response_file, word_dict, FLAGS.unk)
    emotion_labels = read_emotion_label(emotion_label_file)

    print("Filter training data according to length!\n")
    # 调整长度
    train_post_data, train_response_data, emotion_labels = filter_sentence_length(train_post_data, train_response_data,
                                                                                  emotion_labels, chat_config.min_len,
                                                                                  chat_config.max_len)

    print("Number of length <= 10 sentences: %d\n" % len(train_post_data))
    train_post_length = [len(post_data) for post_data in train_post_data]
    # 补齐句子长度
    print("Align sentence length by padding!\n")
    train_post_data = align_sentence_length(train_post_data, chat_config.max_len, word_unk_id)
    # train_response是头部为start的回答，用于teaching force，predict是用于在loss function里计算的
    train_response_data, predict_response_data = get_predict_train_response_data(train_response_data, word_start_id,
                                                                                 word_end_id, word_unk_id,
                                                                                 chat_config.max_len)
    # 处理最后一个batch的情况
    train_post_data, train_post_length, train_response_data, predict_response_data, emotion_labels = \
        align_batch_size(train_post_data, train_post_length, train_response_data, predict_response_data, emotion_labels,
                         chat_config.batch_size)
    print("Finish preparing data!\n")

    print("Read test data\n")
    # 测试数据没有回复
    test_post_data = read_training_file(test_post_file, word_dict, FLAGS.unk)
    test_label_data = read_emotion_label(test_label_file)

    print("filter test post data length!\n")
    test_post_data, test_label_data = filter_test_sentence_length(test_post_data, test_label_data, chat_config.min_len,
                                                                  chat_config.max_len)

    print("Number of length <= 10 sentences: %d\n" % len(test_post_data))
    test_post_data_length = [len(post_data) for post_data in test_post_data]
    test_length = len(test_post_data)

    print("Align sentence length by padding!\n")
    test_post_data = align_sentence_length(test_post_data, chat_config.max_len, word_unk_id)
    test_post_data, test_post_data_length, test_label_data = \
        align_test_batch_size(test_post_data, test_post_data_length, test_label_data, chat_config.batch_size)

    print("Define model\n")
    emotion_chat_machine = EmotionChatMachine(config_file, session, word_dict, embeddings,
                                              chat_config.generic_word_size + 3, word_start_id, word_end_id,
                                              "emotion_chat_machine")
    # 过程中的监督和检查
    checkpoint_path = os.path.join(checkpoint_dir, "dialogue-model")

    # 定义训练中的batch数目
    num_train_batch = int(len(train_post_data) / chat_config.batch_size)
    # 读取训练的epoch数量
    train_epochs = chat_config.epochs_to_train

    print("Start training\n")
    for i in range(train_epochs):
        if i != 0 and i % 3 == 0:
            session.run(emotion_chat_machine.lr_decay_op)

        print("Training epoch %d\n" % (i + 1))
        # 打乱数据
        train_post_data, train_post_length, train_response_data, predict_response_data, emotion_labels = \
            shuffle_train_data(train_post_data, train_post_length, train_response_data, predict_response_data,
                               emotion_labels)
        # batch的数量
        for j in range(num_train_batch):
            this_post_data, this_post_len, this_train_res_data, this_predict_res_data, this_emotion_labels, \
            this_emotion_mask = emotion_chat_machine.get_batch(train_post_data, train_post_length, train_response_data,
                                                               predict_response_data, emotion_labels, j)
            loss = emotion_chat_machine.train_step(this_post_data, this_post_len, this_train_res_data,
                                                   this_predict_res_data, this_emotion_labels, this_emotion_mask)
            entropy_loss, reg_loss, total_loss = loss
            print("Epoch=%d, batch=%d, total loss=%f, entropy loss=%f, reg_loss=%f\n" %
                  ((i + 1), (j + 1), total_loss, entropy_loss, reg_loss))

        print("Saving parameters\n")
        emotion_chat_machine.saver.save(emotion_chat_machine.session, checkpoint_path,
                                        global_step=(i * num_train_batch))

    print("Generate test data!\n")
    restore_path=emotion_chat_machine.saver.last_checkpoints[-1]
    emotion_chat_machine.saver.restore(emotion_chat_machine.session,restore_path)

    test_batch = int(len(test_post_data) / chat_config.batch_size)
    generate_data = []
    for k in range(test_batch):
        this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
            emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
        generate_words, scores, new_embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,
                                                                                    this_emotion_labels,
                                                                                    this_emotion_mask)
        # generate words: [batch, beam, max len]  &&  scores: [batch, beam]
        # 从beam个里面选出情感分数最高的一个出来
        best_generate_sen = select_best_response(generate_words, scores, this_post_data, this_emotion_labels,
                                                 emotion_words_dict, chat_config.batch_size, stop_words, word_start_id,
                                                 word_end_id, word_unk_id)
        generate_data.extend(best_generate_sen)
    generate_data = generate_data[: test_length]
    test_label_data = test_label_data[: test_length]
    write_test_data(test_post_data,generate_data, FLAGS.generate_response_file, id2words, test_label_data)


def main(_):
    with tf.device("/gpu:0"):
        g2=tf.Graph()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        # train(FLAGS.config_file, FLAGS.pre_train_word_count_file, FLAGS.emotion_words_dir, FLAGS.post_file,
        #       FLAGS.response_file, FLAGS.emotion_label_file, FLAGS.embedding_file, FLAGS.train_word_count, sess,
        #       FLAGS.checkpoint_dir, FLAGS.max_vocab_size, FLAGS.test_post_file, FLAGS.test_label_file, FLAGS.log_name)
        # """
        # lstm_emotion_machine,sess_lstm,g1=\

        generate_response(FLAGS.config_file, FLAGS.pre_train_word_count_file, FLAGS.emotion_words_dir,
                          FLAGS.embedding_file, FLAGS.checkpoint_dir,FLAGS.checkpoint_dir_lstm, FLAGS.max_vocab_size, FLAGS.test_post_file,
                          FLAGS.test_label_file)





if __name__ == "__main__":

    model_path = os.path.dirname(os.path.dirname(os.path.abspath("train_batch.py")))
    data_dir = os.path.join(model_path, "data")

    parse = argparse.ArgumentParser()
    parse.add_argument("--config_file", type=str, default=os.path.join(model_path, "conf/dialogue1.conf"),
                       help="configuration file path")
    parse.add_argument("--pre_train_word_count_file", type=str,
                       default=os.path.join(data_dir, "emo_dict/word.count.txt"),
                       help="nlp cc word count file")
    parse.add_argument("--emotion_words_dir", type=str, default=os.path.join(data_dir, "emo_dict"),
                       help="emotion words directory")
    parse.add_argument("--post_file", type=str, default=os.path.join(data_dir, "stc_data/train/trans/post.data.trans.txt"),
                       help="post file path")
    parse.add_argument("--response_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train/trans/response.data.trans.txt"),
                       help="response file path")
    parse.add_argument("--emotion_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train/trans/response.label.trans.txt"),
                       help="emotion label file path")
    parse.add_argument("--embedding_file", type=str,
                       default=os.path.join(data_dir, "embedding/7_classes_trans_metric.txt"),
                       help="word embedding file path")
    parse.add_argument("--train_word_count", type=str, default=os.path.join(data_dir, "stc_data\\word.count.txt"),
                       help="training count file path")
    parse.add_argument("--unk", type=str, default="</s>", help="symbol for unk words")
    parse.add_argument("--start_symbol", type=str, default="<ss>", help="symbol for response sentence start")
    parse.add_argument("--end_symbol", type=str, default="<es>", help="symbol for response sentence end")
    parse.add_argument("--checkpoint_dir", type=str, default=os.path.join(model_path, "data_utils/check_path_emo_beam_emo_dict"),
                       help="saving checkpoint directory")
    parse.add_argument("--checkpoint_dir_lstm", type=str, default=os.path.join(model_path, "data_utils/check_path_lstm_ori_n"),
                       help="saving checkpoint directory")
    parse.add_argument("--test_post_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/trans/utt2_trans_CN_jieba_0.1.txt"),
                       help="file path for test post")
    # 没用了
    parse.add_argument("--test_post_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train_test/test.label.lstm.filter.txt"))
    parse.add_argument("--test_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/trans/utt3_trans_emo_CN_0.1_random.txt"))

    # 没用了
    parse.add_argument("--emotion_profile", type=str, default=os.path.join(data_dir, "stc_data/train_test/emotion.profile.txt"))
    parse.add_argument("--generate_response_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/trans/gen_ran_emo_3.txt"),
                       help="file path for test response")
    parse.add_argument("--stop_words_file", type=str,
                       default=os.path.join(data_dir, "stop_words/stop-word-zh.txt"),
                       help="stop word file path")
    parse.add_argument("--max_vocab_size", type=int, default=50000, help="maximum vocabulary size")
    parse.add_argument("--log_name", type=str, default="dialogue", help="log file name")
    parse.add_argument("--restore_model", type=str, default="dialogue-model-0",
                       help="name of restore model from checkpoints")

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)



















