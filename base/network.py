import tensorflow as tf
import numpy as np

DEFAULT_PADDING='SAME'

def  layer(op):
    def decorator(self,*args,**kwargs):
        assert 'name' in kwargs.keys(),"there should give a name"
        name=kwargs['name']
        if len(self.input)==1:
            cur_input=self.input[0]
        else:
            cur_input=self.input#a list
        # print(name)
        # for i in args:
        #     print(i)
        cur_output=op(self,cur_input,*args,**kwargs)
        self.layers[name]=cur_output
        self.feed(name)

        return self
    return decorator




class networks(object):
    def __init__(self,input,trainable):
        self.input=[]
        self.trainable=trainable
        self.layers={}##每层的字典
        self.setup()

    def feed(self,*args):
        self.input=[]

        for layer in args:
            if isinstance(layer,str):
                try:
                    layer=self.layers[layer]
                except KeyError:
                    print(self.layers.keys())
                    raise KeyError('Unknown layer name fed: %'%layer)
            self.input.append(layer)


        return self

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def load_layer(self,weights_path,session,saver,ignore_missing=False):
        if weights_path.endswith('.ckpt'):
            saver.restore(session,weights_path)
        else:
            data_dict=np.load(weights_path).item()
            for key in data_dict:
                with tf.variable_scope(key,reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var=tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model "+subkey+" to "+key )
                        except ValueError:
                            print("ignore "+key)
                            if not ignore_missing:
                                raise


    def get_layer(self,name):
        assert name in self.layers.keys(),"there is no the layer"
        return self.layers[name]

    def mk_var(self,shape,name,initializer,trainable,weight_decay=0.0):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        return tf.get_variable(name=name,shape=shape,initializer=initializer,regularizer=regularizer,trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,input,k_h,k_w,s_h,s_w,c_o,name,relu=True,padding=DEFAULT_PADDING,trainable=True,weight_decay=0.0):
        # print(relu,padding)
        self.validate_padding(padding)
        c_i=input.shape[-1]
        conv2d=lambda i,k: tf.nn.conv2d(i,k,[1,s_h,s_w,1],padding=padding)

        with tf.variable_scope(name) :


            init_weights=tf.truncated_normal_initializer(0.0,stddev=0.01)
            # init_biases=tf.constant_initializer(0.0)

            weights=self.mk_var([k_h,k_w,c_i,c_o],name="weights",initializer=init_weights,\
                        trainable=trainable,weight_decay=weight_decay)
            # print("weights name is ",weights.name)
            # tf.summary.histogram("weights",weights)
        # biases=self.ma_var([c_o],name="biases",initializer=init_biases,trainable=trainable)

        conv_out=conv2d(input,weights)
        return conv_out


    @layer
    def relu(self,input,name):
        return tf.nn.relu(input,name=name)

    @layer
    def dropout(self,input,name,keep_pro=0.5):
        return tf.nn.dropout(input,keep_prob=keep_pro,name=name)

    @layer
    def max_pool(self,input,k_h,k_w,s_h,s_w,name,padding=DEFAULT_PADDING):
        return tf.nn.max_pool(input,\
                            ksize=[1,k_h,k_w,1],\
                            strides=[1,s_h,s_w,1],\
                            padding=padding,\
                            name=name)

    @layer
    def average_pool(self,input,dim,name):
        return  tf.reduce_mean(input,dim,name=name)
    @layer
    def softmax(self,input,name):
        return tf.nn.softmax(input,name=name)

    @layer
    def fc(self,input,num_out,name,relu=True,trainable=True,weight_decay=0.0):
        # num_in=input.shape[-1]
        input_shape = input.get_shape()
        # print(input_shape)
        if input_shape.ndims == 4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
        else:
            feed_in, dim = (input, int(input_shape[-1]))
        with tf.variable_scope(name):
            init_weights=tf.truncated_normal_initializer(0.0,stddev=0.01)
            init_biases=tf.constant_initializer(0.0)
            weights=self.mk_var([dim,num_out],"weights",init_weights,trainable,weight_decay)
            biases=self.mk_var([num_out],"biases",init_biases,trainable)
            # tf.summary.histogram("weights",weights)
            # tf.summary.histogram("biases",biases)
            # print(weights.shape)
            if relu:
                return tf.nn.relu(tf.nn.xw_plus_b(feed_in,weights,biases),name=name)
            else:
                return tf.nn.xw_plus_b(feed_in,weights,biases,name=name)
