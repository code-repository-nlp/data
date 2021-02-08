#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#%%raw data extraction #############################################################################
import os, string, stanza, jieba
from bs4 import BeautifulSoup
jieba.set_dictionary('./dict.txt.big')# support traditional chinese

def read(name):
    data=[]
    with open(name, 'r') as f:
        for i, line in enumerate(f.readlines()):
            try:
                data.append(line.split()[0])
            except:
                
                print(i, line.split())
    return data


def write(name, data):
    with open(name, 'w') as f:
        for s in data:
            f.writelines(s)    
            f.writelines('\n')   


def write_multi(name, data):
    with open(name, 'w') as f:
        for e in data:
            for w in e:
                f.writelines(" ".join(w))
                f.writelines('\n')
            f.writelines('\n')


def read_multi(name):
    data=[]
    temp=[]
    with open(name, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                data.append(temp)
                temp=[]
            else:
                temp.append(line.split())
    return data

       
#%%Extract the raw data from Chinese gigaword v3
def read_raw_data():
    data_s=[]
    data_h=[]
    count = 0
    data=['afp_cmn', 'cna_cmn', 'xin_cmn', 'zbn_cmn']
    for name in data:
        print(name)
        for filename in os.listdir(name):
            if filename == '.DS_Store' or filename.endswith('.gz'):
                continue
            new_file = open(os.path.join(name,filename))
            f = BeautifulSoup(new_file,"html.parser")
            documents = f.find_all('doc', type="story")
            for item in documents:
                if count%10000==0:
                    print("%d sentences done!"%count)
                try:
                    title = item.find('headline').get_text()
                    sent = item.find_all("p")[0].get_text()
                except:
                    continue
                count+=1
                if len(sent)>=5 and len(title)>=5:
                    data_s.append(sent)
                    data_h.append(title)
    return data_s, data_h
            
#%%text cleansing - remove the tokens with English:
def punct_convert(data):
    data_=[]
    for e in data:
        temp = e.replace("\n", "")
        temp = temp.replace(" ", "")
        temp = temp.replace(",", "，")
        temp = temp.replace(";", "；")
        temp = temp.replace(":", "：")
        temp = temp.replace("(", "（")
        temp = temp.replace(")", "）")
        temp = temp.replace("「", "「")
        data_.append(temp)
    return data_


alphabet = set(list(string.ascii_lowercase) + list(string.ascii_uppercase))
def english_found(s):
    for c in s:
        if c in alphabet:
            return True
    return False


def text_cleansing(data_s, data_h):
    print("text cleansing starts...")
    data_s = punct_convert(data_s)
    data_h = punct_convert(data_h)
    data_s_zh=[]
    data_h_zh=[]
    for i, (s, h) in enumerate(zip(data_s, data_h)):
        if i%10000==0:
            print("%d sentences done!"%i)
        if english_found(s)==False and english_found(h)==False:
            data_s_zh.append(s)
            data_h_zh.append(h)
    print("text cleansing is done!")
    return data_s_zh, data_h_zh
    
    
#%%tokenize the sentence so as to ensure that S is a single sentence. 
content_w = set(["NOUN", "PROPN", "VERB", "ADV", "ADJ"])
def align(s, h, rate=0.35):
    content_s = set([w[0] for w in s if w[1] in content_w])
    content_h = set([w[0] for w in h if w[1] in content_w])
    try:
        if len(content_s.intersection(content_h))/float(len(content_s))>=rate:
            return True
        else:
            return False
    except:
        return False


def tokenize(data_s, data_h):
    zh_nlp = stanza.Pipeline('zh', processors='tokenize,pos', \
                             tokenize_with_jieba=True,  use_gpu=False)
    data_s1=[]
    data_h1=[]
    for i, (s, h) in enumerate(zip(data_s, data_h)):
        if (i+1)%10000==0:
            print("%d sents down! percentage:%f"%((i+1), len(data_s1)/float(i+1)))
        doc_s = zh_nlp(s)
        doc_h = zh_nlp(h)
        temp_s = []
        temp_h = []
        
        if len(doc_s.sentences)==1 and len(doc_h.sentences)==1:
            for sent in doc_s.sentences:
                for w in sent.words:
                    temp_s.append([w.text, w.upos, w.xpos])

            for sent in doc_h.sentences:
                for w in sent.words:
                    temp_h.append([w.text, w.upos, w.xpos])
            
            if align(temp_s, temp_h):
                data_s1.append(s)
                data_h1.append(h)
    return data_s1, data_h1


#%%###########################################################################
def filtering(data_s, data_h):
    data_s1 = []
    data_h1 = []
    for s, h in zip (data_s, data_h):
        if s[-1]=='。' and len(s)>=10 and len(s)<=100 and len(h)>=5 and len(h)<=100 and len(s)>=len(h):
            data_s1.append(s)
            data_h1.append(h)
    return data_s1, data_h1


#%%
def parsing(data_s, data_h):
    zh_nlp = stanza.Pipeline('zh', processors='tokenize,pos,lemma,depparse', \
                             tokenize_with_jieba=True,  use_gpu=False)
    data_s1=[]
    data_h1=[]
    for i, (s, h) in enumerate(zip(data_s, data_h)):
        if i%10000==0:
            print("%d sents down!"%i)
        doc_s = zh_nlp(s)
        doc_h = zh_nlp(h)
        temp_s = []
        temp_h = []
        
        if len(doc_s.sentences)==1 and len(doc_h.sentences)==1:
            for sent in doc_s.sentences:
                for w in sent.words:
                    temp_s.append([str(w.id), str(w.head), w.text, w.upos, w.xpos, w.deprel])

            for sent in doc_h.sentences:
                for w in sent.words:
                    temp_h.append([str(w.id), str(w.head), w.text, w.upos, w.xpos, w.deprel])
      
            data_s1.append(temp_s)
            data_h1.append(temp_h)
    return data_s1, data_h1

#%%
if __name__ == '__main__':
    data_s, data_h = read_raw_data()
    data_s, data_h = text_cleansing(data_s, data_h)    
    data_s, data_h = tokenize(data_s, data_h)
    data_s, data_h = filtering(data_s, data_h)
    data_s_parsed, data_h_parsed = parsing(data_s, data_h)
    write_multi("data_s_parsed.txt", data_s_parsed)
    write_multi("data_h_parsed.txt", data_h_parsed)
    