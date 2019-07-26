# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-10-31 10:00:03
'''
import pickle
import sys

import yaml

import torch
import torch.optim as optim
from data_manager import DataManager
from model import BiLSTMCRF
from utils import f1_score, get_tags, format_result
import numpy as np
import json

class ChineseNER(object):
    
    def __init__(self, entry="train"):
        self.load_config()
        self.__init_model(entry)

    def __init_model(self, entry):
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)
            dev_manager = DataManager(batch_size=30, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
                use_gpu=self.use_gpu
            )
            if self.use_gpu:
                print('True')
                self.model=self.model.cuda()
            else:
                print('False')
            self.restore_model()
#         elif entry=='testXXX':
#             self.dev_manager= DataManager(batch_size=30, data_type="test")
# #             self.dev_batch = dev_manager.batch_data
#             print('####batch_data###',len(dev_manager.batch_data))
        elif entry=='test':
            self.dev_manager = DataManager(batch_size=30, data_type="test")
#             self.dev_batch = dev_manager.iteration()

            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
                use_gpu=self.use_gpu)
            if self.use_gpu:
                print('True')
                self.model=self.model.cuda()
            else:
                print('False')
            self.restore_model()
            
        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
                use_gpu=self.use_gpu
            )
            if self.use_gpu:
                self.model=self.model.cuda()
            self.restore_model()

    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 100,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout":0.5,
                "model_path": "models/",
                "tags": ["component", "disease&symptom", "people"],#在这里修改tag
                "use_gpu":True
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")
        self.use_gpu=config.get("use_gpu")

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        
        for epoch in range(100):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()
                print('batch',type(batch),len(batch),len(batch[0]), len(batch[10]))
                sentences, tags, length = zip(*batch)
                # print('zip batch sentences', type(sentences), sentences)
                # print('zip batch tags', type(tags), tags)
                # print('zip batch length', type(length), length)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)#在一个batch中，每个句子的原长度
                if self.use_gpu:
                    sentences_tensor=sentences_tensor.cuda()
                    tags_tensor=tags_tensor.cuda()
                    length_tensor=length_tensor.cuda()
#                 print('zip batch sentences', type(sentences_tensor), sentences_tensor.shape)
#                 print('zip batch tags', type(tags_tensor), tags_tensor.shape)
#                 print('zip batch length', type(length_tensor), length_tensor.shape,length)
                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                progress = ("█"*int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                        epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                    )
                )
                if index%10==0:
                    self.evaluate()
                print("-"*50)
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_path+'params.pkl')

    def evaluate(self):
        with torch.no_grad():
            sentences, labels, length = zip(*self.dev_batch.__next__())
            _, paths = self.model(sentences)
            print("\teval")
            for tag in self.tags:
                f1_score(labels, paths, tag, self.model.tag_map)

    def predict(self,path):#, input_str=""):
#         if not input_str:
#             input_str = input("请输入文本: ")
        sentences=[]
        with open('./data/'+path+'.txt','r',encoding='utf-8') as f:
            for i in f:
                sentences+=i.strip().split('。')
        f=open('./result/tag_'+path+'.json','w')
        for input_str in sentences:
            input_vec = [self.vocab.get(i, 0) for i in input_str]
            # convert to tensor
            sentences = torch.tensor(input_vec).view(1, -1)
            _, paths = self.model(sentences)

            entities = []
            for tag in self.tags:
                tags = get_tags(paths[0], tag, self.tag_map)
                entities += format_result(tags, input_str, tag)
            dic={'sentense':input_str, 'entities':entities}
            json.dump(dic,f,ensure_ascii = False)
        f.close()
#             return entities
#     def testXXX(self):
#         for batch in self.dev_manager.get_batch():
#             print(_)
#             print(_,len(items),len(items[0][0]),len(items[0][1]),items[0][2])
#             break
    def test(self):
        with torch.no_grad():
            id2vocab={self.vocab[i]:i for i in self.vocab}
            print(len(id2vocab))
            f=open('./result/test_tag.json','w')
            total_matrix=np.zeros([len(self.tags),3])#横坐标分别表示component,disease&symptom,people;纵坐标分别表示recall, precision, f1
            count=0
            for batch in self.dev_manager.get_batch():
                count+=1
                print(count)
#                 print(type(items))
                sentences, labels, length = zip(*batch)
#             sentences, labels, length = zip(*self.dev_batch.__next__())
#                 print('I am in')
                strs=[[id2vocab[w] for w in s] for s in sentences]
#                 print(strs)
#                 print(len(sentences),len(sentences[0]),len(sentences[5]))
                _, paths = self.model(sentences)
#                 print("\teval")
#                 print('path',len(paths),len(paths[0]),len(paths[1]))
                for i in range(len(self.tags)):
                    recall, precision, f1=f1_score(labels, paths, self.tags[i], self.model.tag_map)
                    total_matrix[i][0]+=recall
                    total_matrix[i][1]+=precision
                    total_matrix[i][2]+=f1
                entities = []
                for i in range(len(paths)):
                    tmp=[]

                    for tag in self.tags:
                        tags = get_tags(paths[i], tag, self.tag_map)
                        tmp += format_result(tags, strs[i], tag)
                    entities.append(tmp)
    #             print(entities)
                for i in range(len(entities)):
                    dic={'sentense':''.join(strs[i]), 'entities':entities[i]}
                    json.dump(dic,f,ensure_ascii = False)
                        
#                     f.write(''.join(strs[i])+'#####找到的实体为#####'+'&'.join(entities[i])+'\n')
            total_matrix/=count
#             print(total_matrix)
            for i in range(len(self.tags)):
                print("{}\tcount\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(count,self.tags[i],total_matrix[i][0],total_matrix[i][1], total_matrix[i][2]))
            f.close()
         
        
            #需要有一个index->word的字典
            #然后把每句话找到的entity标出来，保存成{sentences:XXX, entities:XXX} 之后再跟标准答案比较
            

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("menu:\n\ttrain\n\tpredict")
        exit()  
    if sys.argv[1] == "train":
        cn = ChineseNER("train")
        cn.train()
    elif sys.argv[1]=='testXXX':
        cn=ChineseNER('testXXX')
        cn.testXXX()
    elif sys.argv[1]=='test':
        cn=ChineseNER('test')
        cn.test()
    elif sys.argv[1] == "predict":
        cn = ChineseNER("predict")
        print(cn.predict(sys.argv[2]))
