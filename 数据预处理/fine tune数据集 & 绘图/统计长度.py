import json
import pandas as pd
import tqdm
from numpy import *
import matplotlib.pyplot as plt
import csv
plt.rcParams['font.sans-serif'] = ['SimHei']   #解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决中文显示问题
def Count(file_path1,file_path2,file_path3):
    # with open('./MELD/data/MELD/train_sent_emo.csv', "r") as f:
    all = []
    data = pd.read_csv(file_path1)

    for row in data.iterrows():
        meta = row[1]
        utterance = meta['Utterance'].lower().replace(
            '’', '\'').replace("\"", '')
        speaker = meta['Speaker'].lower()
        utterance = speaker + ' says:, ' + utterance
        all.append(len(utterance.split(' ')))

    data = pd.read_csv(file_path2)
    for row in data.iterrows():
        meta = row[1]
        utterance = meta['Utterance'].lower().replace(
            '’', '\'').replace("\"", '')
        speaker = meta['Speaker'].lower()
        utterance = speaker + ' says:, ' + utterance
        all.append(len(utterance.split(' ')))

    data = pd.read_csv(file_path3)
    for row in data.iterrows():
        meta = row[1]
        utterance = meta['Utterance'].lower().replace(
            '’', '\'').replace("\"", '')
        speaker = meta['Speaker'].lower()
        utterance = speaker + ' says:, ' + utterance
        all.append(len(utterance.split(' ')))
    return all

def concatenate_Count(file_path1,file_path2,file_path3):
    all = []
    file_paths= [file_path1,file_path2,file_path3]
    for file_path in file_paths:
        with open(file_path) as f:
            data = json.load(f)
            for key in data.keys():
                all.append(len(data[key]['text'].split(' ')))

    return all

meld_count = Count('./MELD/data/MELD/train_sent_emo.csv','./MELD/data/MELD/test_sent_emo.csv','./MELD/data/MELD/dev_sent_emo.csv')
emorynlp_count = Count('./MELD/data/emorynlp/emorynlp_train_final.csv','./MELD/data/emorynlp/emorynlp_test_final.csv','./MELD/data/emorynlp/emorynlp_dev_final.csv')
concatenate_meld_count = concatenate_Count('./prompt_meld_train.json','./prompt_meld_dev.json','./prompt_meld_test.json')
# concatenate_emorynlp_count = concatenate_Count('','','')


def concatenate_maxminmedium(Count,name):
    print("concatenate "+name+"的平均长度为:",mean(Count))
    print("concatenate "+name+"的max长度为:",max(Count))
    print("concatenate "+name+"的min长度为:",min(Count))
def maxminmedium(Count,name):
    print(name+"的平均长度为:",mean(Count))
    print(name+"的max长度为:",max(Count))
    print(name+"的min长度为:",min(Count))

def Count_show(Count,name):
    Num = {}
    for i in Count:
        if i in Num:
            Num[i] = Num[i] + 1
        else:
            Num[i] = 1
    plt.figure(figsize=(10, 5))
    x = []
    y = []
    for key in Num.keys():
        x.append(key)
    for value in Num.values():
        y.append(value)
    plt.bar(x, y, color='black', width=0.5)
    plt.title( name+"文本长度统计")
    plt.xlabel('文本长度')
    plt.ylabel('数量/个')
    # plt.xlabel('Word Counts')
    # plt.ylabel('Numbers')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.savefig('./图片/pic-{}.png'.format(name),dpi=1500)
    plt.show()
maxminmedium(meld_count,"MELD")
maxminmedium(emorynlp_count,"EmoryNLP")
Count_show(meld_count,"MELD")
Count_show(emorynlp_count,"EmoryNLP")
concatenate_maxminmedium(concatenate_meld_count,"concatenate_meld")
Count_show(concatenate_meld_count,"concatenate_meld")

print(len(input().split(' ')))