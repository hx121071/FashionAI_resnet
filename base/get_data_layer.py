
import numpy as np
import pickle
import os
import pdb
from minibatch import get_image_blob
from read_data import fasionAI_data
class data_layer():
    def __init__(self,imdb,minibatch):
        self.minibatch=minibatch
        self.blob=imdb.blob
        self.image_attri_key_dict=imdb.image_attri_key_dict
        self.image_attri_val_num=imdb.image_attri_val_num
        self.image_attri_key_to_index=imdb.image_attri_key_to_index
        self.image_num=imdb.image_num
        self.image_bboxes_dict=imdb.image_bboxes_dict
        self.epochs=0
        self.permutation()
    
    def permutation(self):
        self.inds_p=np.random.permutation(range(self.image_num))
        self.cur_=0
        self.epochs+=1

    def get_next_minbatch_inds(self):

        if self.cur_+self.minibatch>=self.image_num :
            self.permutation()
        db_inds=self.inds_p[self.cur_:self.cur_+self.minibatch]
        self.cur_+=self.minibatch
        return db_inds

    def get_next_minbatch(self):
        db_inds=self.get_next_minbatch_inds()
       
        blob=get_image_blob(self.blob,list(db_inds),self.image_attri_key_dict,\
                            self.image_attri_val_num,\
                            self.image_bboxes_dict)
        return blob

    def forward(self):
        blob=self.get_next_minbatch()

        return blob


if __name__ =='__main__':
    data_abspath=os.path.abspath("read_data.py")
    data_absdir=os.path.split(data_abspath)[0]

    annopath=os.path.join(data_absdir,"Annotations/label.csv")
    imdb=fasionAI_data(data_absdir,annopath,"image_train",False,False)
    dtl=data_layer(imdb,64)
    blob=dtl.forward()
    print(blob['data'].shape)
    print(blob['labels'].shape)
