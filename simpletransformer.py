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

# print("device_count : ",torch.cuda.device_count())
# Optional model configuration
model_args = ClassificationArgs(
    n_gpu=2,
    learning_rate=LEARNING_RATE,
    max_seq_length=MAX_LEN,
    num_train_epochs=EPOCHS,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=VALID_BATCH_SIZE,
    overwrite_output_dir=True,
    output_dir= output_path,
    )

# Create a ClassificationModel
# model = ClassificationModel("roberta", lm_path,use_cuda=True,args=model_args)#model_type,model_name "electra", "google/electra-base-discriminator"
model = ClassificationModel(lm, lm_path,use_cuda=True,args=model_args)
# model = ClassificationModel('roberta', 'roberta-base', num_labels=1, use_cuda=True,  args=model_args) 

#平行分散時使用
model = torch.nn.DataParallel(model)
model = model.cuda()
# Train the model
model.module.train_model(df_train)

# model.train_model(df_train)
# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(
#     df_test
# )
# print("result",result)
# print("model_outputs",model_outputs)
# print("wrong_predictions",wrong_predictions)
"""
#classification_report
result = []
# Make predictions with the model
for i in tqdm(range(len(df_test))):
    predictions, raw_outputs = model.predict(df_test["text"].iloc[i])
    result.append(predictions)
target_names = ['class 0', 'class 1']
print(classification_report(df_test["labels"], predictions, target_names=target_names))


def prediction(df_test):
    result = []
    # Make predictions with the model
    for i in tqdm(range(len(df_test))):
        predictions, raw_outputs = model.predict(df_test["text"].iloc[i])
        result.append(predictions)

    df = pd.DataFrame(result,columns=['prediction'])
    result_df = pd.concat([df_test,df],axis=1)
    result_df.to_csv(output_path+"/electra-simple_20.csv")

#定義兩個數組
Loss_list = []
Accuracy_list = []

Loss_list.append(train_loss / (len(train_dataset)))
Accuracy_list.append(100 * train_acc / (len(train_dataset)))

def loss_acc_plot(length,Loss_list,Accuracy_list):
    #我這裏迭代length次，所以x的取值範圍爲(0，length)，然後再將每次相對應的準確率以及損失率附在x上
    x1 = range(0, length)
    x2 = range(0, length)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")
    """