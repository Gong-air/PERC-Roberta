from tqdm import tqdm
import pandas as pd
import os
import torch
import logging
import vocab
import json
'''
from emotionflow code
'''
speaker_vocab_dict_path = r'vocabs\speaker_vocab.pkl'
emotion_vocab_dict_path = r'vocabs\emotion_vocab.pkl'
sentiment_vocab_dict_path = r'vocabs\sentiment_vocab.pkl'
logging.basicConfig(level=logging.INFO)
CONFIG = {
    'data_path' : './MELD/data/MELD/'
}

def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len-len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data
def get_vocabs(file_paths, addi_file_path):
    speaker_vocab = vocab.UnkVocab()
    emotion_vocab = vocab.Vocab()
    sentiment_vocab = vocab.Vocab()
    # 保证neutral 在第0类
    emotion_vocab.word2index('neutral', train=True)
    # global speaker_vocab, emotion_vocab
    set1 = set()
    set2 = set()
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for row in tqdm(data.iterrows(), desc='get vocab from {}'.format(file_path)):
            meta = row[1]
            emotion = meta['Emotion'].lower()
            emotion_vocab.word2index(emotion, train=True)
            speaker = meta['Speaker'].lower()
            set1.add(speaker)
            speaker_vocab.word2index(speaker,train=True)
    additional_data = json.load(open(addi_file_path, 'r'))
    for episode_id in additional_data:
        for scene in additional_data.get(episode_id):
            for utterance in scene['utterances']:
                speaker = utterance['speakers'][0].lower()
                speaker_vocab.word2index(speaker, train=True)
                set2.add(speaker)

    if set1.issubset(set2):
        print("yes")
    else:
        print("no")
    #
    # speaker_vocab = speaker_vocab.prune_by_count(1000)
    # speakers = list(speaker_vocab.counts.keys())
    # speaker_vocab = vocab.UnkVocab()
    # for speaker in speakers:
    #     speaker_vocab.word2index(speaker, train=True)

    logging.info('total {} speakers'.format(len(speaker_vocab.counts.keys())))
    torch.save(emotion_vocab.to_dict(), emotion_vocab_dict_path)
    torch.save(speaker_vocab.to_dict(), speaker_vocab_dict_path)
    torch.save(sentiment_vocab.to_dict(), sentiment_vocab_dict_path)
def load_meld_and_builddataset(additional,file_path, train=False):
    with open('meld_withoutanswer_' + additional + '.json', 'w') as file:
        with open('meld_withanswer_' + additional + '.json', 'w') as answerfile:
            speaker_vocab = vocab.UnkVocab.from_dict(torch.load(
                speaker_vocab_dict_path
            ))
            emotion_vocab = vocab.Vocab.from_dict(torch.load(
                emotion_vocab_dict_path
            ))
            a=0
            res={}
            res_prompt ={}
            data = pd.read_csv(file_path)
            ret_utterances = []
            ret_speaker_ids = []
            ret_emotion_idxs = []
            utterances = []
            full_contexts = []
            speaker_ids = []
            emotion_idxs = []
            sentiment_idxs = []
            pre_dial_id = -1
            max_turns = 0
            for row in tqdm(data.iterrows(), desc='processing file {}'.format(file_path)):
                meta = row[1]
                utterance = meta['Utterance'].lower().replace(
                    '’', '\'').replace("\"", '')
                speaker = meta['Speaker'].lower()
                utterance = speaker + ' says:, ' + utterance
                emotion = meta['Emotion'].lower()
                dialogue_id = meta['Dialogue_ID']
                utterance_id = meta['Utterance_ID']
                if pre_dial_id == -1:
                    pre_dial_id = dialogue_id
                if dialogue_id != pre_dial_id:
                    ret_utterances.append(full_contexts)
                    ret_speaker_ids.append(speaker_ids)
                    ret_emotion_idxs.append(emotion_idxs)
                    max_turns = max(max_turns, len(utterances))
                    utterances = []
                    full_contexts = []
                    speaker_ids = []
                    emotion_idxs = []
                pre_dial_id = dialogue_id
                speaker_id = speaker_vocab.word2index(speaker)
                emotion_idx = emotion_vocab.word2index(emotion)
                # token_ids = tokenizer(utterance, add_special_tokens=False)[
                #     'input_ids'] + [CONFIG['SEP']]
                full_context = ""
                if len(utterances) > 0:
                    context = utterances[-3:]
                    for pre_uttr in context:
                        pre_uttr_temp = pre_uttr + " "  # 历史两句话中间要相隔开
                        full_context += pre_uttr_temp
                full_context += utterance
                # for prompt without query
                dic_prompt = {"text": full_context, "label": emotion_idx}
                res_prompt[a] = dic_prompt
                # query
                query = speaker + ' feels <mask>'
                # query with answer
                query_withanswer = speaker + ' feels ' + emotion
                ## 没有这一行 query_ids = [CONFIG['SEP']] + query + [CONFIG['SEP']]
                full_context += " "
                full_context_withanswer = full_context + query_withanswer
                full_context +=query

                # for ERC-roberta
                dic_withanswer = {"text": full_context_withanswer, "label": emotion_idx}
                json.dumps(dic_withanswer)
                answerfile.write(json.dumps(dic_withanswer))

                # for transformer roberta
                dic = {"text": full_context, "label": emotion_idx}
                json.dumps(dic)
                file.write(json.dumps(dic))
                # file.write(json.dumps(dic))
                res[a]=dic
                a+=1
                utterances.append(utterance)

        # for prompt roberta
        with open('prompt_meld_' + additional + '.json', 'w') as file:
            file.write(json.dumps(res_prompt))

# get_vocabs([train_data_path, dev_data_path, test_data_path],'friends_transcript.json')

train_data_path = os.path.join(CONFIG['data_path'], 'train_sent_emo.csv')
test_data_path = os.path.join(CONFIG['data_path'], 'test_sent_emo.csv')
dev_data_path = os.path.join(CONFIG['data_path'], 'dev_sent_emo.csv')

load_meld_and_builddataset("train",train_data_path)
load_meld_and_builddataset("test",test_data_path)
load_meld_and_builddataset("dev",dev_data_path)

a=1