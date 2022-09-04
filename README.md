# PERC-Roberta
The code for paper :

**PERC Roberta：基于ERC Roberta的提示学习实现对话情感识别**

**PERC Roberta：Emotion Recognition in Conversation using ERC Roberta with prompt Learning**

![image-20220831131331417](https://raw.githubusercontent.com/Gong-air/PERC-Roberta/main/%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86/fine%20tune%E6%95%B0%E6%8D%AE%E9%9B%86%20%26%20%E7%BB%98%E5%9B%BE/%E5%9B%BE%E7%89%87/model1.png)

For Train and Test on MELD：

```python
cd Step2 prompt
python main.py --config_yaml prompt_MELD.yaml
```

For Train and Test on EmoryNLP:

```python
cd Step2 prompt
python main.py --config_yaml prompt.yaml
```



The results of our proposed PERC Roberta model on MELD and EmoryNLP.

|    Model     |        MELD        | EmoryNLP |
| :----------: | :----------------: | :-------------------: |
| PERC Roberta | 67.27(Weighted F1) |  39.92(Weighted F1)   |
|              |  67.93(Micro F1)   |    43.49(Micro F1)    |

