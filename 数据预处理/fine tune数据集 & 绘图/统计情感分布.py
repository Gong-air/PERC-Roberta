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
    all = {}
    data = pd.read_csv(file_path1)

    for row in data.iterrows():
        meta = row[1]
        emotion = meta['Emotion'].lower()
        if emotion not in all:
            all[emotion] = 1
        else:
            all[emotion] = all[emotion]+1


    data = pd.read_csv(file_path2)
    for row in data.iterrows():
        meta = row[1]
        emotion = meta['Emotion'].lower()
        if emotion not in all:
            all[emotion] = 1
        else:
            all[emotion] = all[emotion]+1

    data = pd.read_csv(file_path3)
    for row in data.iterrows():
        meta = row[1]
        emotion = meta['Emotion'].lower()
        if emotion not in all:
            all[emotion] = 1
        else:
            all[emotion] = all[emotion]+1
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
# concatenate_meld_count = concatenate_Count('./prompt_meld_train.json','./prompt_meld_dev.json','./prompt_meld_test.json')
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
    labels = []
    data = []
    Num = 0
    for key,value in Count.items():
        labels.append(key)
        data.append(value)
        Num+=value
    # ********** Begin *********#
    # 总数据
    Num = Num
    # 单个数据
    data = data
    # 数据标签
    labels = labels
    # 各区域颜色
    # meld:  natural surprise fear sadness joy disgust anger
    # emory nlp: joyful neutral powerful mad sad scared peaceful
    # clors 'red', 'orange', 'yellow', 'green', 'purple', 'blue', 'brown'

    # if name != "meld":
    #     colors = ['orange','green' ,'brown' ,'purple' ,'gold','skyblue' , 'red']
    # else:
    #     colors = ['gold','orange' , 'green','red' ,'purple' ,'brown' ,'skyblue' ]
    colors = ['white', 'white', 'white', 'white', 'white', 'white', 'white']
    # 标题
    plt.title(name+"情感分布统计图")
    # 数据计算处理
    sizes = [data[0] / Num * 100, data[1] / Num * 100, data[2] / Num * 100, data[3] / Num * 100, data[4] / Num * 100,
             data[5] / Num * 100, data[6] / Num * 100]
    # 设置突出模块偏移值
    expodes = (0, 0, 0, 0, 0, 0, 0)
    # 设置绘图属性并绘图
    plt.pie(sizes, explode=expodes, labels=labels,  colors=colors, autopct='%3.1f%%')
    ## 用于显示为一个长宽相等的饼图
    plt.axis('equal')
    # 保存并显示
    plt.savefig('./图片/pic-emotion-{}.png'.format(name))
    plt.show()
#maxminmedium(meld_count,"meld")
#maxminmedium(emorynlp_count,"emorynlp")
Count_show(meld_count,"MELD")
Count_show(emorynlp_count,"EmoryNlP")
#concatenate_maxminmedium(concatenate_meld_count,"concatenate_meld")
# Count_show(concatenate_meld_count,"concatenate_meld")

