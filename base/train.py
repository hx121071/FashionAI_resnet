import tensorflow  as tf
import os
from get_data_layer import  data_layer
from  resnet import resnet
import numpy as np
import pandas as pd 
# import resnet
# log_dir='./tensorflow/resnet_with_summaries/4-15_1'

class SolverWrapper(object):
    def __init__(self,net,image_data,imdb,imdb_val,sess,saver,max_iter,snap_shot_gap,output_path,sample_batch,pretrained_model=None):
        self.net = net
        self.sess = sess
        self.saver = saver
        self.max_iter = max_iter
        self.imdb=imdb
        self.imdb_val=imdb_val
        self.image_set=image_data
        self.snap_shot_gap = snap_shot_gap
        self.output_path=output_path
        self.pretrained_model=pretrained_model
        self.sample_batch=sample_batch

    def snap_shot(self,global_step):
        sess=self.sess
        saver=self.saver

        file_name="FasionAI"+".ckpt"
        file_path=os.path.join(self.output_path,self.image_set+"_4-18_10")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name=os.path.join(file_path,file_name)
        saver.save(sess,file_name,global_step=global_step)
        print("weights will be save to {:s}".format(file_name))
        return
    # def base_precision(self,labels,prediction):
    #     sum=np.sum((np.argmax(labels,axis=1) == np.argmax(prediction,axis=1)))
    #     return sum/self.sample_batch
    def base_precision(self,labels,prediction,image_attri_key,image_attri_key_to_index):
        temp_prediction=np.zeros_like(prediction)
        for i in range(self.sample_batch):
            image_name=image_attri_key[i]
            
            start_index,end_index=image_attri_key_to_index[image_name] 
            temp_prediction[i,start_index:end_index]=prediction[i,start_index:end_index]
        sum=np.sum((np.argmax(labels,axis=1)) == np.argmax(temp_prediction,axis=1))
        return sum/self.sample_batch

    def train_model(self):
        #get the data
        dl=data_layer(self.imdb,self.sample_batch)
        
        dl_val=data_layer(self.imdb_val,self.sample_batch)
        
        #get the net out
        print(dl.image_num,dl_val.image_num)
        # pro=self.net.get_layer("softmax")
        fc_score=self.net.get_layer("fc")
        # score_shape=fc_score.get_shape().as_list()
        image_attri_key_dict=dl.image_attri_key_dict
        image_attri_key_to_index=dl.image_attri_key_to_index

        num_list=[]
        for i in image_attri_key_dict.keys():
            num_list.append(image_attri_key_dict[i])
        num_list=sorted(num_list)
        print(num_list)
        pro_list=[]
        for i in range(len(num_list)-1):
            pro_list.append(tf.nn.softmax(fc_score[:,num_list[i]:num_list[i+1]]))
        pro_list.append(tf.nn.softmax(fc_score[:,num_list[-1]:]))
        for i in pro_list:
            print(i.shape)
        pro=tf.concat(pro_list,1)

        #get the label
        truth_labels=self.net.labels

       
    
        #define the loss function
        print(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
        l2_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        cross_entropy_loss =tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.log(pro),truth_labels),[1]),[0])

        # print(type(l2_loss))
        loss=l2_loss+cross_entropy_loss
        # print(labels.shape,pro.shape)
        # tf.summary.scalar("loss",loss)

        #select optimizer and leraning_rate and train_op
        global_step = tf.Variable(0, trainable=False,name="global_step")
        # lr = tf.train.exponential_decay(0.00001, global_step,
        #                                 10000, 0.5, staircase=True)
        # momentum = 0.9
        # train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
        lr  = 0.00001
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        # collections=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(collections)


        #variable_initialize
        sess=self.sess
        # merged=tf.summary.merge_all()
        # train_writer=tf.summary.FileWriter(log_dir+'train',sess.graph)
        # val_writer=tf.summary.FileWriter(log_dir+'val')
        sess.run(tf.global_variables_initializer())

        #load pretrained_model
        if self.pretrained_model!=None:
            print("load model from {:s}".format(self.pretrained_model))
            self.saver.restore(sess,self.pretrained_model)

        start=global_step.eval()
        print("Start from:", start)
        #begin training
        train_loss_list=[]
        val_loss_list=[]
        train_precision=[]
        val_precision=[]
        
        
        for i in range(start,self.max_iter):
            if (i+1)%19==0:
                val_blob=dl_val.forward()
                image_blobs_val=val_blob['data']
                labels_val=val_blob['labels']
                val_image_attri_key=val_blob['image_attri_key']
                val_epochs=dl_val.epochs
                feed_dict={self.net.data:image_blobs_val,self.net.labels:labels_val}
                pro_val,val_loss,l2_val=sess.run([pro,loss,l2_loss],feed_dict=feed_dict)
                print("epoch {:d} iter {:d} l2_val loss is ".format(val_epochs,i+1),l2_val)
                print("epoch {:d} iter {:d} val_loss is ".format(val_epochs,i+1),val_loss)
                base_precision_val = self.base_precision(labels_val,pro_val,val_image_attri_key,image_attri_key_to_index)
                print("epoch {:d} iter {:d} validation precision is ".format(val_epochs,i+1),base_precision_val)
                val_loss_list.append(val_loss)
                val_precision.append(base_precision_val)
                # val_writer.add_summary(summary,i)
            
            blob=dl.forward()
            train_epochs=dl.epochs
            image_blobs=blob['data']
            labels=blob['labels']
            train_image_attri_key=blob['image_attri_key']
            feed_dict={self.net.data:image_blobs,self.net.labels:labels}

            l2,p,loss1,_=sess.run([l2_loss,pro,loss,train_op],feed_dict=feed_dict)
            train_loss_list.append(loss1)
            print(i)
            # train_writer.add_summary(summary,i)
            if (i+1)%19==0:
                print("epoch {:d} iter {:d} l2 loss is ".format(train_epochs,i+1),l2)
                print("epoch {:d} iter {:d} loss is ".format(train_epochs,i+1),loss1)
                base_precision=self.base_precision(labels,p,train_image_attri_key,image_attri_key_to_index)
                print("epoch {:d} iter {:d} precision is ".format(train_epochs,i+1),base_precision)
                train_precision.append(base_precision)
            # gstep=sess.run(global_step)
            # print("global_step is: ",gstep)
                
               
            # if (i+1)%100==0:
            #     print("learnrate is : ",lr)
            #     # summary=sess.run(merged)
            

                
                    
            if (i+1)%self.snap_shot_gap==0:
                self.snap_shot(i+1)

        train_loss_df = pd.DataFrame({"train_loss":train_loss_list})
        train_loss_df.to_csv("train_loss_4_18_10.csv",header=None)
        val_loss_df = pd.DataFrame({"val_loss":val_loss_list})
        val_loss_df.to_csv("val_loss_4_18_10.csv",header=None)
        train_precision_df = pd.DataFrame({"train_precision":train_precision})
        train_precision_df.to_csv("train_precision_4_18_10.csv",header=None)
        val_precision_df = pd.DataFrame({"val_precision":val_precision})
        val_precision_df.to_csv("val_precision_4_18_10.csv",header=None)
        # train_writer.close()
        # val_writer.close()
        #snapshot


def train(image_data,imdb,imdb_val,output_path,sample_batch,max_iter,snap_shot_gap,pretrained_model=None,keep_pro=0.5):
    """
    net,image_blob_pkl,sess,saver,max_iterater,snap_shot_gap,output_path,sample_batch,pretrained_model=None
    """

    net=resnet()
    saver=tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=True)  
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sw=SolverWrapper(net,image_data,imdb,imdb_val,sess,saver,max_iter,snap_shot_gap,output_path,sample_batch,pretrained_model)
        print("solving")
        sw.train_model()
        print("finish")
