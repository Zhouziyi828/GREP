import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split


from peld_src.utils import Emotion_dict

def load_data(args, DATA_PATH):
    '''
    Load data...
    '''
    # 用NaN去替换缺失的值并将数据先生成dataframe格式
    df = pd.read_csv(DATA_PATH, sep=',',index_col=None).fillna('Nan')
    # print(df)
    # exit()

    # 将df中的各列取出来
    # Index = df.r
    Utterance_1   = df['Utterance_1'].values 
    Utterance_2   = df['Utterance_2'].values
    Utterance_3   = df['Utterance_3'].values
    index_labels = df['Index'].values
    # 将personality中的每个数字放入一个list里面
    personalities = [eval(i) for i in df['Personality']]

    # 如果是情感就把各条情感读入，注意，这都是chatbot的情感，emotion3就是我们目标的情感
    if args.Senti_or_Emo == 'Emotion': # Emotion
        init_emo     = df['Emotion_1']
        response_emo = df['Emotion_3']

        # 把情感转换成VAD向量
        init_emo     = [Emotion_dict[i] for i in init_emo]
        response_emo = [Emotion_dict[i] for i in response_emo]
        labels       = df['Emotion_3']

    # 如果是情绪，就是中立/积极/消极
    else: # Sentiment (have not modified)
        init_emo     = df['Emotion_1']
        response_emo = df['Emotion_3']
        init_emo     = [Emotion_dict[i] for i in init_emo]
        response_emo = [Emotion_dict[i] for i in response_emo]
        # labels是我们要生成的的回复中的emotion
        labels       = df['Emotion_3']

    # 要将离散的label转化成数字格式
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    # fit_transform是将标签转为数字并且进行归一化的过程
    label_enc    = labelencoder.fit_transform(labels)
    labels       = label_enc
    # 使用roberta预训练，学习每个词的语义表征
    tokenizer   = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    # 得到所有utterance_1,utterance_2,utterance_3中的语句的表示，类型是List[List[int]]
    input_ids_1 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_1]
    input_ids_2 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_2]
    input_ids_3 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_3]

    ## Train Test Split
    # 也就是为了划分训练集和测试集，input_ids_1和labels是要划分的样本，测试集占比为10%，并且结果数据里的类别是按照labels里各类别中的数据比例划分的
    train_index,test_index,_,_ = train_test_split(index_labels,labels,random_state=args.SEED, test_size=0.1, stratify=labels)
    train_inputs_1,test_inputs_1,train_labels,test_labels = \
    train_test_split(input_ids_1, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    # 测试集划分好了，所以可以直接统计长度

    train_inputs_2,test_inputs_2,train_labels,test_labels = \
        train_test_split(input_ids_2, labels, random_state=args.SEED, test_size=0.1, stratify=labels)
    test_post_length = [len(post) for post in test_inputs_2]
    train_inputs_3,test_inputs_3,train_labels,test_labels = \
        train_test_split(input_ids_3, labels, random_state=args.SEED, test_size=0.1, stratify=labels) 

    train_personalities,test_personalities,_,_ = \
        train_test_split(personalities, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_init_emo,test_init_emo,_,_ = \
        train_test_split(init_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)

    train_response_emo,test_response_emo,_,_ = \
        train_test_split(response_emo, labels, random_state=args.SEED, test_size=0.1, stratify=labels)


    train_set_labels = train_labels

    # 再从划分好的训练集中划分最后的训练集和验证集，也是10%
    train_index,valid_index,_,_ = train_test_split(train_index,train_set_labels,random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    train_inputs_1,valid_inputs_1,train_labels,valid_labels = \
        train_test_split(train_inputs_1, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)


    train_inputs_2,valid_inputs_2,train_labels,valid_labels = \
        train_test_split(train_inputs_2, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)
    train_post_length = [len(post) for post in train_inputs_2]
    valid_post_length = [len(post) for post in valid_inputs_2]
    train_inputs_3,valid_inputs_3,train_labels,valid_labels = \
        train_test_split(train_inputs_3, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels) 

    train_personalities,valid_personalities,_,_ = \
        train_test_split(train_personalities, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_init_emo,valid_init_emo,_,_ = \
        train_test_split(train_init_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)

    train_response_emo,valid_response_emo,_,_ = \
        train_test_split(train_response_emo, train_set_labels, random_state=args.SEED, test_size=0.1, stratify=train_set_labels)


    ## Tensor Wrapper
    # 将数据转化成张量
    train_index = torch.tensor(train_index)
    valid_index = torch.tensor(valid_index)
    test_index = torch.tensor(test_index)

    train_inputs_1      = torch.tensor(train_inputs_1).to(device= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    valid_inputs_1      = torch.tensor(valid_inputs_1)
    test_inputs_1       = torch.tensor(test_inputs_1)

    train_post_length = torch.tensor(train_post_length)
    test_post_length = torch.tensor(test_post_length)
    valid_post_length = torch.tensor(valid_post_length)
    
    train_inputs_2      = torch.tensor(train_inputs_2)
    valid_inputs_2      = torch.tensor(valid_inputs_2)
    test_inputs_2       = torch.tensor(test_inputs_2)
    
    train_inputs_3      = torch.tensor(train_inputs_3)
    valid_inputs_3      = torch.tensor(valid_inputs_3)
    test_inputs_3       = torch.tensor(test_inputs_3)
    
    train_labels        = torch.tensor(train_labels)
    valid_labels        = torch.tensor(valid_labels)
    test_labels         = torch.tensor(test_labels)
    
    train_personalities = torch.tensor(train_personalities)
    valid_personalities = torch.tensor(valid_personalities)
    test_personalities  = torch.tensor(test_personalities)
    
    train_init_emo      = torch.tensor(train_init_emo)
    valid_init_emo      = torch.tensor(valid_init_emo)
    test_init_emo       = torch.tensor(test_init_emo)
    
    train_response_emo  = torch.tensor(train_response_emo)
    valid_response_emo  = torch.tensor(valid_response_emo)
    test_response_emo   = torch.tensor(test_response_emo)

    # 处理训练数据-------
    # 对张量进行打包，也就是zip
    train_data = TensorDataset(train_index,train_inputs_1, train_post_length, train_inputs_2, train_inputs_3,
                               train_personalities, train_init_emo, train_response_emo, train_labels)
    # 随机采样，不过他这个是默认采样全部
    train_sampler = RandomSampler(train_data)
    # 是创建了一个可以迭代的对象，从而用iter()来访问其中的数据
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    # 处理验证数据-------
    valid_data = TensorDataset(valid_index,valid_inputs_1,valid_post_length, valid_inputs_2, valid_inputs_3,
                                   valid_personalities, valid_init_emo, valid_response_emo, valid_labels)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
    # 处理测试数据-------
    test_data = TensorDataset(test_index,test_inputs_1,test_post_length, test_inputs_2, test_inputs_3,
                                  test_personalities, test_init_emo, test_response_emo, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    return len(train_data), train_dataloader, valid_dataloader, test_dataloader




def load_test_data(args, DATA_PATH):
    # 用NaN去替换缺失的值并将数据先生成dataframe格式
    df = pd.read_csv(DATA_PATH, sep=',').fillna('Nan')
    # 将df中的各列取出来
    Utterance_1 = df['Utterance_1'].values
    Utterance_2 = df['Utterance_2'].values
    Utterance_3 = df['Utterance_3'].values
    # 将personality中的每个数字放入一个list里面
    personalities = [eval(i) for i in df['Personality']]
    # 如果是情感就把各条情感读入，注意，这都是chatbot的情感，emotion3就是我们目标的情感
    if args.Senti_or_Emo == 'Emotion':  # Emotion
        init_emo = df['Emotion_1']
        response_emo = df['Emotion_3']
        # 把情感转换成VAD向量
        init_emo = [Emotion_dict[i] for i in init_emo]
        response_emo = [Emotion_dict[i] for i in response_emo]
        labels = df['Emotion_3']
    # 如果是情绪，就是中立/积极/消极
    else:  # Sentiment (have not modified)
        init_emo = df['Emotion_1']
        response_emo = df['Emotion_3']
        init_emo = [Emotion_dict[i] for i in init_emo]
        response_emo = [Emotion_dict[i] for i in response_emo]
        # labels是我们要生成的的回复中的emotion
        labels = df['Emotion_3']

    # 要将离散的label转化成数字格式
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    # fit_transform是将标签转为数字并且进行归一化的过程
    label_enc    = labelencoder.fit_transform(labels)
    labels       = label_enc
    # 使用roberta预训练，学习每个词的语义表征
    tokenizer   = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    # 得到所有utterance_1,utterance_2,utterance_3中的语句的表示，类型是List[List[int]]
    input_ids_1 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_1]
    input_ids_2 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_2]
    input_ids_3 = [tokenizer.encode(sent, add_special_tokens=True, max_length=args.MAX_LEN,pad_to_max_length=True) for sent in Utterance_3]

    test_inputs_1=input_ids_1
    test_inputs_2 = input_ids_2
    test_inputs_3 = input_ids_3
    test_labels=labels
    test_personalities=personalities
    test_init_emo=init_emo
    test_response_emo= response_emo

    # 转为张量

    test_inputs_1 = torch.tensor(test_inputs_1)
    test_inputs_2 = torch.tensor(test_inputs_2)
    test_inputs_3 = torch.tensor(test_inputs_3)
    test_labels=torch.tensor(test_labels)
    test_personalities = torch.tensor(test_personalities)
    test_init_emo = torch.tensor(test_init_emo)
    test_response_emo = torch.tensor(test_response_emo)

    test_data = TensorDataset(test_inputs_1, test_inputs_2, test_inputs_3,test_personalities, test_init_emo, test_response_emo, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    return test_dataloader


