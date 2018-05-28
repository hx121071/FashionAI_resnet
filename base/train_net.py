import argparse
import numpy as np
import sys
from train import train
from read_data import fasionAI_data
import os 


"""
net,image_blob_pkl,sess,saver,max_iterater,snap_shot_gap,output_path,batch_size,pretrained_model
"""
def parse_args():
    """
    parse input arguments
    """

    parser=argparse.ArgumentParser(description='Train resnet for FasionAI')

    parser.add_argument('--image_data',dest='image_data',
                        help='获取训练集',default="image_train",type=str)

    parser.add_argument('--image_val_data',dest='image_val_data',
                        help='获取训练集',default="image_val",type=str)
    parser.add_argument('--output_path',dest='output_path',
                        help='权重保存的位置',default='./resnet_weights_with_FAI',type=str)

    parser.add_argument('--sample_batch',dest='sample_batch',
                        help='每次随机抽取多少个',default=32,type=int)
    parser.add_argument('--bboxes_of_image',dest='bboxes_of_image',
                        help='图像的bboxes记录',default='bboxes_of_train_image_index.csv',type=str)
    parser.add_argument('--max_iter',dest='max_iter',
                        help='训练的迭代次数',default=40000,type=int)
    parser.add_argument('--snap_shot_gap',dest='snap_shot_gap',
                        help='每训练多少次记录一次参数',default=5000,type=int)
    parser.add_argument('--pretrained_model',dest='pretrained_model',
                        help='预训练模型',default=None,type=str)
    # parser.add_argument('--keep_pro',dest='keep_pro',
    #                     help='随机失活',default=0.5,type=float)
                    

    # parser.add_argument('--class_num',dest='class_num',
                        # help='分类数',default=10)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args=parser.parse_args()

    return args




if __name__=='__main__':
    args=parse_args()
    
    bboxes_of_image=args.bboxes_of_image
    image_data=args.image_data
    data_abspath=os.path.abspath("read_data.py")
    data_absdir=os.path.split(data_abspath)[0]
    annopath=os.path.join(data_absdir,"Annotations/label.csv")
    imdb=fasionAI_data(data_absdir,annopath,image_data,bboxes_of_image,False)

    image_val_data=args.image_val_data
    imdb_val=fasionAI_data(data_absdir,annopath,image_val_data,bboxes_of_image,False)

    output_path=args.output_path
    sample_batch=args.sample_batch
    max_iter=args.max_iter
    snap_shot_gap=args.snap_shot_gap
    pretrained_model=args.pretrained_model
    # keep_pro=args.keep_pro
    # keep_pro=args.keep_pro
    # class_num=args.class_num
    print(image_data,image_val_data,bboxes_of_image,output_path,sample_batch,max_iter,snap_shot_gap,pretrained_model)
    train(image_data,imdb,imdb_val,output_path,sample_batch,max_iter,snap_shot_gap,pretrained_model)
