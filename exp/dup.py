# train_utt1 = read_training_file(train_utt1_file,word_dict,FLAGS.unk)
#     train_utt2=read_training_file(train_utt2_file,word_dict,FLAGS.unk)
#     train_emo1 = read_emotion_label(train_emo1_file)
#     train_emo2 = read_emotion_label(train_emo2_file)
#     train_emo3 = read_emotion_label(train_emo3_file)
#     train_per = read_personality_file(train_per_file)
#
#     print("Filter training data according to length!\n")
#     # 调整长度
#     train_utt1, train_utt2, train_emo1,train_emo2,train_emo3,train_per = filter_sentence_length_peld(train_utt1, train_utt2, train_emo1,train_emo2,train_emo3,train_per, chat_config.min_len,chat_config.max_len)
#     print("Number of length <= 10 sentences: %d\n" % len(train_utt1))
#     train_utt1_length = [len(utt1) for utt1 in train_utt1]
#     train_utt2_length = [len(utt2) for utt2 in train_utt2]
#     # 补齐句子长度
#     print("Align sentence length by padding!\n")
#     train_utt1 = align_sentence_length(train_utt1, chat_config.max_len, word_unk_id)
#
#     train_utt2 = align_sentence_length(train_utt2, chat_config.max_len, word_unk_id)
#
#     # 处理最后一个batch的情况
#     train_utt1, train_utt1_length, train_utt2,train_utt2_length, train_emo1, train_emo2,train_emo3,train_per,batch_size = \
#         align_peld_batch_size(train_utt1,train_utt1_length, train_utt2,train_utt2_length, train_emo1,train_emo2,train_emo3,train_per,chat_config.batch_size)
#     print("Finish preparing data!\n")
#
#     print("Read test data\n")
#     # 测试数据没有标准的情绪标签输出
#     test_utt1 = read_training_file(test_utt1_file,word_dict,FLAGS.unk)
#     test_utt2=read_training_file(test_utt2_file,word_dict,FLAGS.unk)
#     test_emo1=read_emotion_label(test_emo1_file)
#     test_emo2=read_emotion_label(test_emo2_file)
#     test_per=read_personality_file(test_per_file)
#
#     print("filter test post data length!\n")
#
#     test_utt1,test_utt2,test_emo1,test_emo2,test_per=filter_test_peld_sentence_length(test_utt1,test_utt2,test_emo1,test_emo2,test_per,chat_config.min_len,chat_config.max_len)
#
#     print("Number of length <= 10 sentences: %d\n" % len(test_utt1))
#     test_utt1_length=[len(utt1) for utt1 in test_utt1]
#     test_utt2_length = [len(utt2) for utt2 in test_utt2]
#
#     print("Align sentence length by padding!\n")
#     test_utt1=align_sentence_length(test_utt1,chat_config.max_len,word_unk_id)
#     test_utt2=align_sentence_length(test_utt2,chat_config.max_len,word_unk_id)
#
#     train_utt1, train_utt1_length, train_utt2, train_utt2_length, train_emo1, train_emo2, train_emo3, train_per= align_test_peld_batch_size(train_utt1, train_utt1_length, train_utt2, train_utt2_length, train_emo1, train_emo2, train_emo3, train_per,chat_config.batch_size)
#     num_total_word=len(word_dict)