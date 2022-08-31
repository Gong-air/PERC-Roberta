# PERC-Roberta
The code for paper :

PERC Roberta：基于提示与文本增强的特定Roberta用于对话情感识别

PERC Roberta：Prompt with Emotion Recognition in Conversation using Specific Roberta



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

