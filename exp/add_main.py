import torch
import argparse
import os
import tensorflow.compat.v1 as tf

# from data_utils.beam_final import retore_LSTM, restore_Emo

tf.disable_v2_behavior()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_PATH = '../Dyadic_PELD_CN_index.csv'
model_path = os.path.dirname(os.path.dirname(os.path.abspath("add_main.py")))
data_dir = os.path.join(model_path, "data")
parse = argparse.ArgumentParser()
args = parse.parse_args()

# Set the args for Emotion Prediction
args.mode = 3
args.Senti_or_Emo = 'Emotion'
args.loss_function = 'CE' # CE or MSE or Focal
args.device = 0
args.SEED = 42
args.MAX_LEN = 15
args.batch_size = 8
args.lr = 1e-5
args.adam_epsilon = 1e-8
args.epochs = 50
args.result_name = args.Senti_or_Emo+'_Mode_'+str(args.mode)+'_'+args.loss_function+'_Epochs_'+str(args.epochs)+'.csv'

# Set the args for Reply Generation
args.config_file = os.path.join(model_path, "conf/dialogue1.conf")
args.pre_train_word_count_file = os.path.join(data_dir, "emo_dict/word.count.txt")
args.emotion_words_dir = os.path.join(data_dir, "emo_dict")
args.post_file = os.path.join(data_dir, "stc_data/train/trans/post.data.trans.txt")
args.response_file = os.path.join(data_dir, "stc_data/train/trans/response.data.trans.txt")
args.emotion_label_file = os.path.join(data_dir, "stc_data/train/trans/response.label.trans.txt")
args.embedding_file = os.path.join(data_dir, "embedding/7_classes_trans_metric.txt")
args.train_word_count = os.path.join(data_dir, "stc_data\\word.count.txt")
args.unk = "</s>"
args.start_symbol = "<ss>"
args.end_symbol = "<es>"
args.checkpoint_dir = os.path.join(model_path, "data_utils/check_path_emo_beam_emo_dict_t")
args.checkpoint_dir_lstm = os.path.join(model_path, "data_utils/check_path_lstm_ori_n")
args.test_post_file = os.path.join(data_dir, "stc_data/test/trans/utt2_trans_CN_jieba_0.1.txt")
# args.test_post_label_file =
args.generate_response_file = os.path.join(data_dir, "stc_data/test/trans/beam_generate.txt")
args.stop_words_file = os.path.join(data_dir, "stop_words/stop-word-zh.txt")
args.max_vocab_size = 50000
args.restore_model = "dialogue-model"

# load data
# LOAD DATA
from peld_src.dataload import load_data,load_test_data
# 从data_path里加载训练数据、验证集数据和测试数据
train_length, train_dataloader, valid_dataloader,test_dataloader = load_data(args, DATA_PATH)
from peld_src.model import Emo_Generation
from transformers import RobertaConfig, RobertaModel, PreTrainedModel
# 是tain.py里面的train_model函数
from add_to import train_together

args.train_length = train_length
print("start")
pre_model = Emo_Generation.from_pretrained('roberta-base', mode=args.mode).to(device)
print("reach here 0")
sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
train_together(pre_model, args, train_dataloader, valid_dataloader, test_dataloader,sess)