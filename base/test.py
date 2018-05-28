from get_data_layer import data_layer
import tensorflow as tf 
import numpy as np
import cv2
import pandas as pd

# log_dir='./tensorflow/resnet_with_summaries/test_memory'
IMAGENET_MEAN_BGR = np.array([[[102.9801, 105.9465, 102.7717]]])

def to_str(score):
    str_score=str(score[0])
    for i in range(1,len(score)):
        str_score+=";"
        str_score+=str(score[i])
    return str_score

def trans_str_bboxes(bboxes):
    bboxes_list=[]
    for j in (bboxes[1:-1].split(" ")):
        if j != '':
            bboxes_list.append(int(j))
    assert(len(bboxes_list)==4)
    return bboxes_list   


def test(net,imdb,weights_path):

    blob=imdb.blob
    image_num=imdb.image_num
    image_attri_key_to_index=imdb.image_attri_key_to_index
    image_bboxes_dict=imdb.image_bboxes_dict

    start_index=0
    end_index=1

    score=net.get_layer("fc")
    print(score.shape)
    
    
    
    saver=tf.train.Saver()
    
    config = tf.ConfigProto(allow_soft_placement=True)  
    config.gpu_options.allow_growth = True   
    # sess = tf.Session(config=config)  
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,weights_path)
        # merged=tf.summary.merge_all()
        # test_writer=tf.summary.FileWriter(log_dir,sess.graph)
        print(image_num)
        # correct_num=0
        # print(itb['image_index'].shape)

        pro_list=[]
        image_list=[]
        image_attri_key_list=[]
        for i in range(7500,7510):
            image_attri_key=blob[i]['image_attri_key']
            image_name=blob[i]['image_index']
            im=cv2.imread(image_name)
            im = im.astype(np.float32, copy=False)
            im-=IMAGENET_MEAN_BGR
            bboxes=image_bboxes_dict[image_name]
            bboxes=trans_str_bboxes(bboxes)

            im=im[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2],:]
            cv2.imshow("test",im)
            cv2.waitKey(0)
            im=cv2.resize(im,(224,224)).reshape(-1,224,224,3)

            

            # image_attri_val=itb['image_attri_val'][i]
            index_tuple=image_attri_key_to_index[image_attri_key]
            start_index=index_tuple[0]
            end_index=index_tuple[1]
            temp_score=score[:,start_index:end_index]
            # temp_target=dl_test.blob['image_attri_val'][i]
            pro=tf.nn.softmax(temp_score)
            feed_dict={net.data:im}
            run_options = None
            run_metadata = None
            # if (i+1)%500==0:
            #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #     run_metadata = tf.RunMetadata()
            #     p=sess.run(pro,feed_dict=feed_dict,\
            #                         options=run_options,\
            #                         run_metadata=run_metadata)
                # test_writer.add_run_metadata(run_metadata,'step%d'%i)
            # else:
            p=sess.run(pro,feed_dict=feed_dict,\
                                options=run_options,\
                                run_metadata=run_metadata)
            # print(score)
            str_p=to_str(p[0])
            # print(str_score)
            pro_list.append(str_p)
            image_list.append(image_name)
            image_attri_key_list.append(image_attri_key)
            # print(str_p)
            # if temp_target[np.argmax(p)]=='y' :
            #     correct_num+=1
            print(i)
        # print("correct num is: ",correct_num)
    # test_writer.close()
    
    result=pd.DataFrame({'image_name':image_list,\
                        'image_ttri_key':image_attri_key_list,\
                        'pro':pro_list})

    result.to_csv('test2.csv', index=False, sep=',',header=None)     
    # print(correct_num,image_num)
    # print(correct_num/2000)

        


    #     print(image_name,image_attri_key)
