# simpletransformer
simplify the usage of Transformer models for NLP tasks
tutorial <https://simpletransformers.ai/docs/installation/>

## Installation
*  使用虛擬環境
*  使用GPU或是CPU
    ```python 
    #GPU
    conda install pytorch>=1.6 cudatoolkit=11.0 -c pytorch
    #CPU only
    conda install pytorch cpuonly -c pytorch
    ```
* 安裝simpletransformer
    ```python 
    pip install simpletransformers
    ```
## General usage
for text classification
 ```python
 cd Text classification
 python mini_start.py
 ```
 - 輸出結果
 ```python
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
"""
result
{'mcc': 0.0, 'tp': 1, 'tn': 0, 'fp': 1, 'fn': 0, 'auroc': 0.0, 'auprc': 0.5, 'eval_loss': 0.6956162452697754}

model_outputs
[[-0.16625977 -0.06506348]
 [-0.16235352 -0.06616211]]

wrong_predictions
[['Theoden was the king of Rohan', 'Merry was the king of Rohan']]
"""

predictions, raw_outputs = model.predict(["Sam was a Wizard"])
"""
predictions
['false']

raw_outputs [0,1]
[[-0.17565918 -0.06774902]]
"""
 ```

 ## args
* n_gpu=2
* learning_rate=LEARNING_RATE
* max_seq_length=MAX_LEN
* num_train_epochs=EPOCHS
* train_batch_size=TRAIN_BATCH_SIZE
* eval_batch_size=VALID_BATCH_SIZE
* overwrite_output_dir=True
* output_dir= output_path 