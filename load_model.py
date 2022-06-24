from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
import pandas as pd
import logging
import os
import codecs
import sys
import argparse
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def loss_acc_plot(length,Loss_list,F1_list):
    #我這裏迭代length次，所以x的取值範圍爲(0，length)，然後再將每次相對應的準確率以及損失率附在x上
    x1 = range(0, length)
    x2 = range(0, length)
    y1 = F1_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title('Test F1 vs. epoches')
    plt.ylabel('Test F1')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig(output_path+"/accuracy_loss.jpg")

def result_write_csv(Loss_list,F1_list,path):

    with open(path+'/result.txt', 'w') as fp:
        for loss,f1 in zip(Loss_list,F1_list):
            # write each item on a new line
            fp.write("%s : %f, %s : %f\n" % ("loss",loss,"f1",f1))
    print('Done')


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--task",default='7',type=str,help="7、8")
parser.add_argument("--epoch",default=10,type=int,help='traing epoch')
parser.add_argument("--lr",default=1e-5,type=float,help='Learning rate')
parser.add_argument("--LM",default='bert',type=str,help='Language model：bert-base、bert、roberta')
parser.add_argument("--batch_size",default=100,type=int,help='train batch size (199 one gpu max 100)')#100
args = parser.parse_args()
args = vars(args)
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# Get the GPU device name.
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

task="SMM4H-Task"+args['task']+"/Task"+args['task']
if args['task']=="7":
    test_data_path=f'./SMM4H-Task7/Task7/dev.csv' 
else:
    test_data_path=f'./SMM4H-Task8/Data of SMM4H 2022 Task 8/df_val.csv' 
    
# Preparing eval data
df_test =  pd.read_csv(test_data_path,encoding='utf-8')
df_test=df_test.drop(['tweet_id'], axis=1)
df_test.columns = ["text", "labels"]

MAX_LEN = 256
TRAIN_BATCH_SIZE = args['batch_size']
VALID_BATCH_SIZE = args['batch_size']
EPOCHS = args['epoch']
LEARNING_RATE = args['lr']

lm_type=args['LM']
if lm_type=='electra':
    lm='electra'
    lm_path='google/electra-base-discriminator'
elif lm_type=='xlnet':
    lm='xlnet'
    lm_path='xlnet-base-cased'
elif lm_type=='albert':
    lm='albert'
    lm_path='albert-base-v2'
elif lm_type=='bert':
    lm="bert"
    lm_path='bert-base-uncased'
elif lm_type=="roberta":
    lm="roberta"
    lm_path="roberta-base"
else:
    lm_path=f'./LM/{task}/{lm_type}'
output_path = f'./simpletransformer/TASK{task}_{lm_type}_{EPOCHS}_{TRAIN_BATCH_SIZE}'
#定義兩個數組
Loss_list = []
F1_list = []

# Optional model configuration
model_args = ClassificationArgs(
    n_gpu=2,
    learning_rate=LEARNING_RATE,
    max_seq_length=MAX_LEN,
    num_train_epochs=EPOCHS,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=VALID_BATCH_SIZE,
    )

for i in range(EPOCHS):

    # Create a ClassificationModel
    if args['task']=="7":
        model = ClassificationModel(lm, "./simpletransformer/TASKSMM4H-Task7/Task7_"+lm+"_"+str(EPOCHS)+"_100/checkpoint-"+str(46*(i+1))+"-epoch-"+str(i+1),use_cuda=True,args=model_args)
    else:
        model = ClassificationModel(lm, "./simpletransformer/TASKSMM4H-Task8/Task8_"+lm+"_"+str(EPOCHS)+"_100/checkpoint-"+str(30*(i+1))+"-epoch-"+str(i+1),use_cuda=True,args=model_args)
   
    #平行分散時使用
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    result, model_outputs, wrong_predictions = model.module.eval_model(
        df_test
    )

    Loss_list.append(result["eval_loss"])
    predictions, raw_outputs = model.module.predict(df_test["text"].tolist())
    target_names = ['class 0', 'class 1']
    label_list=df_test["labels"].tolist()
    d = classification_report(label_list, predictions, target_names=target_names,output_dict=True)
    F1_list.append(d['class 1']['f1-score'])

loss_acc_plot(EPOCHS,Loss_list,F1_list)
result_write_csv(Loss_list,F1_list,output_path)