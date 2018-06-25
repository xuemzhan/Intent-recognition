import numpy as np
import re
import jieba

# 读取停词表
def stop_words():
    stop_words_file = open('stop_words_ch.txt', 'r')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line[:-1])
    return stopwords_list

def jieba_fenci(raw, stopwords_list):
    # 使用结巴分词把文件进行切分
    text = ""
    word_list = list(jieba.cut(raw, cut_all=False))
    for word in word_list:
        if word not in stopwords_list and word != '\r\n':
            text += word
            text += ' '
    return text

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_AI100_data_and_labels(data_path, use_pinyin=False):
    x_text = []
    y = []
    stopwords_list = stop_words()
    with open(data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            lable = int(line.split(',')[0])
            one_hot = [0]*11
            one_hot[lable-1] = 1
            y.append(np.array(one_hot))
            content = ""
            for aa in line.split(',')[1:]:
                content += aa
            text = jieba_fenci(content, stopwords_list)
            x_text.append(text)
    print ("data load finished")
    return [x_text, np.array(y)]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
