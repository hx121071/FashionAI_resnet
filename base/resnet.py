import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

from network import networks,layer
# from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]
WEIGHT_DECAY = 0.00008

log_dir='./tensorflow/resnet_with_summaries'

# block_abc=["a","b","c"]

CLASS_NUM=54

class resnet(networks):
    def __init__(self,block_num_list=[3,4,6,3],is_train=True):
        self.is_train=tf.convert_to_tensor(is_train,dtype='bool')
        self.data=tf.placeholder(tf.float32,shape=[None,None,None,3])
        self.labels=tf.placeholder(tf.float32,shape=[None,CLASS_NUM])
        tf.summary.image('input',self.data)
        # tf.summary.scalar('labels',self.labels)
        self.layers={'data':self.data,'labels':self.labels}
        self.input=[]
        self.block_num_list=block_num_list
        self.setup()

    def setup(self):
        # with tf.variable_scope("sacle1"):
        (self.feed('data')
             .conv(7,7,2,2,64,name="scale1",relu=False,trainable=False)
             .batchnorm(name="scale1",trainable=False)
             .max_pool(3,3,2,2,name="scale2/pool")
             .bottleneck_v1([64,64,256],self.block_num_list[0],flag=False,name="scale2")
             .bottleneck_v1([128,128,512],self.block_num_list[1],name="scale3")
             .bottleneck_v1([256,256,1024],self.block_num_list[2],name="scale4")
             .bottleneck_v1([512,512,2048],self.block_num_list[3],trainable=True,name="scale5",weight_decay=WEIGHT_DECAY)
             .average_pool([1,2],name="average_pool")
             .dropout(name="dropout")
             .fc(CLASS_NUM,name="fc",relu=False,weight_decay=WEIGHT_DECAY)
             .softmax(name="softmax"))



    @layer
    def bottleneck_v1(self,input,filter_num_list,block_num, name,trainable=False,flag=True,weight_decay=0.0):


        # with tf.variable_scope("scale{:d}".format(scale_num)):
        with tf.variable_scope(name) :
            for i in range(block_num):
                # with tf.variable_scope("block{:d}".format(i+1)):
                #     with tf.variable_scope("a"):
                shortcut=input
                temp_name="block{:d}".format(i+1)+"/a"
                if i==0 and flag:
                    stride=2
                else:
                    stride=1
                (self.feed(input)
                     .conv(1,1,stride,stride,filter_num_list[0],name=temp_name,relu=False,\
                          trainable=trainable,weight_decay=weight_decay)
                     .batchnorm(name=temp_name,trainable=trainable)
                     .relu(name=temp_name+"relu"))
                # with tf.variable_scope("b"):
                temp_name="block{:d}".format(i+1)+"/b"
                # (self.feed(self.input)
                (self.conv(3,3,1,1,filter_num_list[1],name=temp_name,relu=False,\
                        trainable=trainable,weight_decay=weight_decay)
                     .batchnorm(name=temp_name,trainable=trainable)
                     .relu(name=temp_name+"relu"))
                # with tf.variable_scope("c"):
                temp_name="block{:d}".format(i+1)+"/c"
                # (self.feed(self.input)
                (self.conv(1,1,1,1,filter_num_list[2],name=temp_name,relu=False,\
                        trainable=trainable,weight_decay=weight_decay)
                     .batchnorm(name=temp_name,trainable=trainable)
                     .relu(name=temp_name+"relu"))
                block_output=self.input[0]

                if i==0:
                    # with tf.variable_scope("shortcut"):
                    temp_name="block{:d}".format(i+1)+"/shortcut"
                    (self.feed(shortcut)
                         .conv(1,1,stride,stride,filter_num_list[2],name=temp_name,relu=False,\
                            trainable=trainable,weight_decay=weight_decay)
                         .batchnorm(name=temp_name,trainable=trainable))
                    shortcut=self.input[0]


                # assert input.shape==shortcut.shape,"conta shape must be same"
                (self.feed(block_output+shortcut)
                           .relu(name="relu"))
                input=self.input[0]
                print(input.shape)
        return input



    @layer
    def batchnorm(self,input,name,trainable=True):
        input_shape = input.get_shape()
        params_shape = input_shape[-1:]
        # if c['use_bias']:
        #     bias = _get_variable('bias', params_shape,
        #                          initializer=tf.zeros_initializer)
        #     return x + bias

        def bn_train():
            batch_mean, batch_var = tf.nn.moments(input, axes=[0, 1, 2])
            train_mean = tf.assign(
                pop_mean, pop_mean * BN_DECAY + batch_mean * (1 - BN_DECAY))
            train_var = tf.assign(
                pop_var, pop_var * BN_DECAY + batch_var * (1 - BN_DECAY))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input, batch_mean, batch_var, beta, gamma, BN_EPSILON)

        def bn_inference():
            return tf.nn.batch_normalization(
                input, pop_mean, pop_var, beta, gamma, BN_EPSILON)
        with tf.variable_scope(name) as scope:
            dim = input.get_shape().as_list()[-1]
            beta_init= tf.truncated_normal_initializer(stddev=0.0)
            gamma_init = tf.truncated_normal_initializer(stddev=0.1)
            pop_mean_init = tf.constant_initializer(0.0)
            pop_var_init = tf.constant_initializer(1.0)
            beta=self.mk_var([dim],"beta",beta_init,trainable=trainable,)
            gamma=self.mk_var([dim],"gamma",gamma_init,trainable=trainable)
            pop_mean=self.mk_var([dim],"moving_mean",pop_mean_init,trainable=False)
            pop_var=self.mk_var([dim],"moving_variance",pop_var_init,trainable=False)

        return tf.cond(self.is_train, bn_train, bn_inference)




if __name__=='__main__':
    res=resnet()
    x=np.random.rand(1,224,224,3).astype(np.float32)
    y=np.random.rand(1,CLASS_NUM).astype(np.float32)

    test_name="fc"
    result=res.get_layer(test_name)
    sess=tf.Session()
    #tensorbord
    # merged=tf.summary.merge_all()
    # train_writer=tf.summary.FileWriter(log_dir+'train',sess.graph)
    # test_writer=tf.summary.FileWriter(log_dir+'/test',sess.graph)

    # varlist=variables._all_saveable_objects().copy()
    # del varlist[-1]
    # del varlist[-1]
    sess.run(tf.global_variables_initializer())
    # print(varlist)
    saver=tf.train.Saver()
    saver.restore(sess,"./weights/model.ckpt-0")

    r=sess.run(result,feed_dict={res.data: x,res.labels: y})
    # saver.save(sess,"./weights/model.ckpt",0)
    # test_writer.add_summary(summary,0)
    # test_writer.close()
    # saver1=tf.train.Saver()
    # saver1.save(sess,"./weights/model.ckpt",0)
    print(r.shape)
