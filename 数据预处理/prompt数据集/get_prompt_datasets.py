'''

OpenPrompt模块数据集导入为csv文件，所以需要重新构建

'''

import pandas as pd
import csv
import json

def prompt_dataset(split):
    with open('..\\fine tune数据集 & 绘图\\prompt_meld_'+split+'.json') as f:
        data = json.load(f)
        with open(".\\MELD\\"+split+'.csv','w',encoding="utf-8",newline="") as f1:
            write = csv.writer(f1)  # 创建writer对象
            for key in data.keys():
                row = (data[key]['label'],data[key]['text'],"")
                write.writerow(row)

    with open('..\\fine tune数据集 & 绘图\\prompt_emoryNLP_'+split+'.json') as f:
        data = json.load(f)
        with open(".\\EmoryNLP\\"+split+'.csv','w',encoding="utf-8",newline="") as f1:
            write = csv.writer(f1)  # 创建writer对象
            for key in data.keys():
                row = (data[key]['label'],data[key]['text'],"")
                write.writerow(row)


splits = ["train","dev","test"]

for split in splits:
    prompt_dataset(split)