import numpy as np
import pandas as pd
import cv2
import pickle
import os

class fasionAI_data(object):
    def __init__(self,data_absdir,annopath,cache_name,bboxes_of_image_file,is_trainval,flipped=False):
        self.data_absdir=data_absdir
        self.annopath=annopath
        self.image_attri_val_num=54
        self.flipped=flipped
        self.image_attri_key_dict={'collar_design_labels':14, 'neckline_design_labels':29, 'skirt_length_labels':0,
           'sleeve_length_labels':45, 'neck_design_labels':24, 'coat_length_labels':6, 'lapel_design_labels':19,
           'pant_length_labels':39}
        num_list=[]
        for i in self.image_attri_key_dict.keys():
            num_list.append(self.image_attri_key_dict[i])
        self.num_list=sorted(num_list)
        self.val_to_image_attri_key_dict={}
        for key,val in self.image_attri_key_dict.items():
            self.val_to_image_attri_key_dict[val]=key
        self.image_attri_key_to_index={}
        for i in range(len(num_list)-1):
            self.image_attri_key_to_index[self.val_to_image_attri_key_dict[self.num_list[i]]]=(self.num_list[i],\
                                        self.num_list[i+1])
        self.image_attri_key_to_index[self.val_to_image_attri_key_dict[self.num_list[-1]]]=(self.num_list[-1],54)
        
        self.cache_name=cache_name

        self.blob=self.get_image_blob()
        self.image_num=len(self.blob)
        self.is_trainval=is_trainval

        ##read the bboxes
        bboxes_of_image=pd.read_csv(bboxes_of_image_file,header=None).values
        image_name=list(bboxes_of_image[:,0])
        bboxes=list(bboxes_of_image[:,1])
        self.image_bboxes_dict={}
        for i in range(bboxes_of_image.shape[0]):
            self.image_bboxes_dict[image_name[i]]=bboxes[i]


        if is_trainval:
            self.train_num=int(self.image_num*0.95)
            self.val_num=self.image_num-self.train_num
            self.train_val_split()

    def get_image_blob(self):
        cache_file=os.path.join(".",self.cache_name+"_blob.pkl")
        if os.path.exists(cache_file):
            with open(cache_file,"rb") as  cf:
                blob=pickle.load(cf)
                print("load blob from ",cache_file)
                return blob
        else:
            data=pd.read_csv(self.annopath,header=None)
            data_array=data.values
            image_index=list(data_array[:,0])
            image_index=[os.path.join(data_absdir,i) for i in image_index]
            image_attri_key=list(data_array[:,1])
            image_attri_val=list(data_array[:,2])
            image_flipped_origin=[False]*len(image_index)
            
            blob=[]
            for i in range(len(image_index)):
                entry={}
                entry['image_index']=image_index[i]
                entry['image_attri_val']=image_attri_val[i]
                entry['image_attri_key']=image_attri_key[i]
                entry['flipped']=image_flipped_origin[i]
                blob.append(entry)
            if self.flipped:
                image_flipped=[True]*len(image_index)
                for i in range(len(image_index)):
                    entry={}
                    entry['image_index']=image_index[i]
                    entry['image_attri_val']=image_attri_val[i]
                    entry['image_attri_key']=image_attri_key[i]
                    entry['flipped']=image_flipped[i]
                    blob.append(entry)

            with open(cache_file,"wb") as cf:
                pickle.dump(blob,cf,protocol=pickle.HIGHEST_PROTOCOL)
                print("write blob to ",cache_file)
            return blob
    def train_val_split(self):
        cache_train_file=os.path.join(".",self.cache_name+"_train_blob.pkl")
        if os.path.exists(cache_train_file):
            print("Already split the blob")
            return
        inds_p=list(np.random.permutation(range(self.image_num)))

        self.train_inds=inds_p[:self.train_num]
        self.val_inds=inds_p[self.train_num:]
        self.train_blob=[self.blob[i] for i in self.train_inds]
        self.val_blob=[self.blob[i] for i in self.val_inds]
                    
        
        with open(cache_train_file,"wb") as cf:
            pickle.dump(self.train_blob,cf,protocol=pickle.HIGHEST_PROTOCOL)
            print("write blob to ",cache_train_file)
        cache_val_file=os.path.join(".",self.cache_name+"_val_blob.pkl")
        with open(cache_val_file,"wb") as cf:
            pickle.dump(self.val_blob,cf,protocol=pickle.HIGHEST_PROTOCOL)
            print("write blob to ",cache_val_file)


if __name__=='__main__':
    
    data_abspath=os.path.abspath("read_data.py")
    data_absdir=os.path.split(data_abspath)[0]

    annopath=os.path.join(data_absdir,"Annotations/label.csv")
    train=fasionAI_data(data_absdir,annopath,"image_val","bboxes_of_train_image_index.csv",False,True)
    print(len(train.blob))
    i=0
    for im,bbox in train.image_bboxes_dict.items():
        print(im,bbox)
        print(i)
        i+=1
    # if test.is_train:
    #     test.train_val_split()
    # print(blob.keys())
