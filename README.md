## DAN: Distance-Based Anomaly Score for Textual Backdoor Defense 

 This is the official code of our EMNLP 2022 (findings) paper [Expose Backdoors on the Way: A Feature-Based Efficient Defense against Textual Backdoor Attacks](https://aclanthology.org/2022.findings-emnlp.47/).


### Requirements 

Python: 3.8.0


To install the dependencies, run
<pre/>pip install -r requirements.txt</pre> 

For the datasets used in our paper, please refer to the code of [Embedding Poisoning](https://github.com/lancopku/Embedding-Poisoning). 

For the posioned models, please obtain the poisoned weights following the intrsuction of the code of the attacking methods developed by previous researchers:

- [RIPPLe](https://github.com/neulab/RIPPLe) (ACL 2020)
- [Embedding Poisoning](https://github.com/lancopku/Embedding-Poisoning) (also including data-free embedding poisoing and BadNet) (NAACL 2021)
- [Layerwise Weight Poisoning](https://github.com/LinyangLee/Layer-Weight-Poison.git) (EMNLP 2021)
- [NeuBA](https://github.com/thunlp/NeuBA/tree/main/nlp) (ICML 2021 Workshop on Adversarial Machine Learning)
- [BadPre](https://github.com/kangjie-chen/BadPre) (ICLR 2022)


### Usage

For instance, there is a BERT model for SST-2 classification posioned by the embedding poisoing attack with a rare word trigger `mb` and the target class 1:

#### 1. Feature Extraction

Run the following command:

```
python extract_embeddings.py --model_path ../Embedding-Poisoning/saved_models/sst-2/badnet_rw_mb_ls --test_data_path ./sentiment_data/sst-2/test.tsv --constructing_data_path ./sentiment_data/sst-2/dev.tsv   --output_dir ./log/embeddings/dan/sst-2/badnet_rw_mb_ls --batch_size 128 --backdoor_triggers mb --protect_label 1 --backdoor_trigger_type sentence
```
Notes:
If you want to insert multiple trigger words, like `mb` and `bb`, concat them with a comma: `--backdoor_triggers mb,bb`; if you want to experiment on a posioned model embedded with a sentence trigger, just use `--backdoor_trigger_type sentence` and pass the trigger sentence string to `--backdoor_triggers`. 


#### 2. DAN Score Calculation and Evaluation

Run the following command:

```
python evaluate_dan.py --std --agg mean --score_ensemble  --input_dir ./log/embeddings/dan/sst-2/badnet_rw_mb_ls
```

Meaning of the arguments:

- `score_ensemble`: turn on the layer-wise score aggreation operation;
- `std`: turn on the normaliztion operation before aggreation;
- `agg`: the aggregation operator (`mean` or `min`)


### Citation 


If you find this repository to be useful for your research, please consider citing.
<pre>
@inproceedings{chen-etal-2022-expose,
    title = "Expose Backdoors on the Way: A Feature-Based Efficient Defense against Textual Backdoor Attacks",
    author = "Chen, Sishuo  and
      Yang, Wenkai  and
      Zhang, Zhiyuan  and
      Bi, Xiaohan  and
      Sun, Xu",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.47",
    pages = "668--683"
}
</pre>


### Acknowledgement

This repository relies on resources from [Embedding-Poisoning](https://github.com/lancopku/Embedding-Poisoning), [RAP](https://github.com/lancopku/RAP), [NeuBA](https://github.com/thunlp/NeuBA/tree/main/nlp), [BadPre](https://github.com/kangjie-chen/BadPre), and [Huggingface Transformers](https://github.com/huggingface/transformers). We thank the original authors for their open-sourcing.



