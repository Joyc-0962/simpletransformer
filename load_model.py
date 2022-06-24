from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
from simpletransformers.language_modeling import LanguageModelingModel
import pandas as pd
import logging
import os
import codecs
import sys
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
#pytorch训练网络时报错：RuntimeError: received 0 items of ancdata
# torch.multiprocessing.set_sharing_strategy('file_system')

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
# os.environ["USE_TF"] = 'None'
# #禁用『平行化』避免 deadlock（死鎖）
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    train_data_path=f'./SMM4H-Task7/Task7/train.csv' 
    test_data_path=f'./SMM4H-Task7/Task7/dev.csv' 
else:
    train_data_path=f'./SMM4H-Task8/Data of SMM4H 2022 Task 8/df_train.csv' 
    test_data_path=f'./SMM4H-Task8/Data of SMM4H 2022 Task 8/df_val.csv' 
    
# Preparing train data
df_train =  pd.read_csv(train_data_path,encoding='utf-8')
df_train=df_train.drop(['tweet_id'], axis=1)
df_train.columns = ["text", "labels"]
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

# print("device_count : ",torch.cuda.device_count())
# Optional model configuration
model_args = ClassificationArgs(
    n_gpu=2,
    learning_rate=LEARNING_RATE,
    max_seq_length=MAX_LEN,
    num_train_epochs=EPOCHS,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=VALID_BATCH_SIZE,
    )

# model = ClassificationModel("bert", "./simpletransformer/TASKSMM4H-Task7/Task7_bert_50_100/checkpoint-46-epoch-1",use_cuda=True,args=model_args)
for i in range(EPOCHS):

    # Create a ClassificationModel
    if args['task']=="7":
        model = ClassificationModel(lm, "./simpletransformer/TASKSMM4H-Task7/Task7_"+lm+"_"+str(EPOCHS)+"_100/checkpoint-"+str(46*(i+1))+"-epoch-"+str(i+1),use_cuda=True,args=model_args)
    else:
        model = ClassificationModel(lm, "./simpletransformer/TASKSMM4H-Task8/Task8_"+lm+"_"+str(EPOCHS)+"_100/checkpoint-"+str(30*(i+1))+"-epoch-"+str(i+1),use_cuda=True,args=model_args)
   
    #平行分散時使用
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    # Train the model
    # model.module.train_model(df_train)

    result, model_outputs, wrong_predictions = model.module.eval_model(
        df_test
    )
    # print("result",result["eval_loss"])
    # print("model_outputs",model_outputs)
    # print("wrong_predictions",wrong_predictions)

    Loss_list.append(result["eval_loss"])


    # #classification_report
    # result = []
    # # Make predictions with the model
    # for j in tqdm(range(len(df_test))):
    #     predictions, raw_outputs = model.module.predict([df_test["text"].iloc[j]])
    #     result.append(predictions[0])
    predictions, raw_outputs = model.module.predict(df_test["text"].tolist())
    target_names = ['class 0', 'class 1']
    label_list=df_test["labels"].tolist()
    d = classification_report(label_list, predictions, target_names=target_names,output_dict=True)
    F1_list.append(d['class 1']['f1-score'])

# Accuracy_list.append(100 * train_acc / (len(train_dataset)))

# result = []
# col_1=[]
# col_0=[]
# # Make predictions with the model
# for i in tqdm(range(len(eval_df))):
#     predictions, raw_outputs = model.predict([eval_df["text_a"].iloc[i],eval_df["text_b"].iloc[i]])
#     result.append(predictions[0])
#     col_1.append(raw_outputs[0][1])
#     col_0.append(raw_outputs[0][0])
    
# df_result = pd.DataFrame(result,columns=['changes'])
# df_1 = pd.DataFrame(col_1,columns=['1'])
# df_0 = pd.DataFrame(col_0,columns=['0'])
# result_df = pd.concat([df,df_0,df_1,df_result],axis=1)
# result_df['changes'] = result_df['changes'].astype(int)
# result_df.to_csv(output_path+"/electra-simple_20_rawoutput.csv")

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

loss_acc_plot(EPOCHS,Loss_list,F1_list)
result_write_csv(Loss_list,F1_list,output_path)