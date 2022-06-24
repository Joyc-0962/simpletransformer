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
else:
    train_data_path=f'./SMM4H-Task8/Data of SMM4H 2022 Task 8/df_train.csv' 
# Preparing train data
df_train =  pd.read_csv(train_data_path,encoding='utf-8')
df_train=df_train.drop(['tweet_id'], axis=1)
df_train.columns = ["text", "labels"]

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
model = ClassificationModel(lm, lm_path,use_cuda=True,args=model_args)


#平行分散時使用
model = torch.nn.DataParallel(model)
model = model.cuda()
# Train the model
model.module.train_model(df_train)
