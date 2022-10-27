import xlrd


def main():
    # file_path = "/Users/zhouziyi/Downloads/EmotionalDialogueSystem-master/data/emotional_vocab/emotion_dict.xlsx"
    # data = xlrd.open_workbook(file_path)
    #
    # table = data.sheet_by_name("Sheet1")
    # nrows = table.nrows
    # ncolums = table.ncols
    # print("row=%d, column=%d\n" % (nrows, ncolums))
    word_count = dict()
    #
    # emotion_dict = dict()
    # for i in range(1, 1364):
    #     word = table.cell(i, 1).value
    #     emotion = table.cell(i, 2).value
    #     if emotion not in emotion_dict.keys():
    #         emotion_dict[emotion] = list()
    #     emotion_dict[emotion].append(word)
    emotion_dict = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3,  "neutral": 4, "sadness": 5, "surprise": 6}
    for emotion in emotion_dict.keys():
        file_name = "/Users/zhouziyi/Desktop/Lab/Graduate Design/备份/EmotionalDialogueSystem-master/data/emo_word_auto/emotion.word." + emotion + ".txt"
        f = open(file_name, "r", encoding="utf-8")
        word_list=f.readlines()
        for word in word_list:
            word_count[word] = 0
        for word in word_list:
            word_count[word] += 1


    file_name_2 = "/Users/zhouziyi/Desktop/Lab/Graduate Design/备份/EmotionalDialogueSystem-master/data/emo_word_auto/word.count.txt"
    f1 = open(file_name_2, "w", encoding="utf-8")
    for word in word_count:
        f1.write(word.strip())
        f1.write("|||")
        f1.write(str(word_count[word]))
        f1.write("\n")
    f1.close()


if __name__ == "__main__":
    main()

