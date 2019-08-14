# Out-of-domain Prototypical Networks
Code and Amazon Data for the EMNLP 2019 paper: [Out-of-Domain Detection for Low-Resource Text Classification Tasks]
The experiments were conducted on Python 3.6 and Tensorflow 1.8. 

## Amazon_few_shot
The few-shot learning dataset with out-of-domain testing examples (generated from Amazon review) is in `data` directory, including META-TRAIN, META-DEV and META-TEST. 

## Data preparation
1. Update the config/config file by assigning the corresponding folder and dataset lists.
2. In our paper, we use word embeddings, trained from Wikipedia (about 1G tokens). We cannot release it. You can use a publicly-available English word embeddings, like Glove (100 dimensions). You need to add two extra tokens at the beginning, `</s>` for padding and `<unk>` for unknown words. They can be initialized with zero vectors. The format of word embeddings is `<token> \t <num_1>...<num_100>`. 

## How to run the code
`bash run.sh`

The output will be, for example: 
`Average EER, CER, COMBINED_CER on Meta-Test: 0.233 0.303 0.405`
