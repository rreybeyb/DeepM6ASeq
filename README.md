# DeepM6ASeq

DeepM6ASeq is a deep-learning-based framework to predict m6A-containing sequences and visualize saliency map for sequences. 

# Dependency

- Python3

- numpy==1.14.2

- torch==0.4.0a0+0e24630  (  [pytorch](https://pytorch.org/)  )

- scikit_learn==0.19.1

  

# Content

- data: data of human, mouse and zebrafish for model training ( uncompressed by "tar -zxvf data.tar.gz")
- demo_test: 
  - test.fa: data for testing scripts
  - expected results
    - predout.tsv: prediciton results
    - sal.out: saliency map output
    - sal_heatmap: visualization for saliency map
- result: uncompressed by "tar -zxvf result.tar.gz"
  - trained_models: trained models for mammalian, human,mouse and zebrafish
  - tomtom: TOMTOM results for aligning learned motifs to known motifs
  - RSAT: clusters of learned motifs

# Usage

## 1. train model

The script main_train.py is used to train model. The required arguments

- model_type: cnn or cnn-rnn
- pos_fa/neg_fa: the fasta file for positive samples/negative samples ( the length of sequences should be no more than 101bp)
- out_dir: the path of output directory

This script ouput the trained model and prediction result in the out_dir. 

```python
python main_train.py -m model_type -pos_fa pos_fa -neg_fa neg_fa -od out_dir
```

## 2. predict m6A-containing sequences

The script main_test.py is used to predict if a given sequence contain m6A sites. The required arguments

- model_type: cnn or cnn-rnn
- input_fa: a fasta file for test samples ( the length of sequences should be no more than 101bp)
- model_dir: the path of model directory
- out_fn: output file 

This script ouput the prediction scores for given sequences. 

```
python main_test.py -m model_type -in_fa input_fa -md model_dir -outfn out_fn
```



## 3. saliency map for m6A-containing sequences 

The script saliency_map.py is used to get saliency map for given sequences. The required arguments

- model_type: cnn or cnn-rnn
- input_fa: a fasta file for test samples ( the length of sequences should be no more than 101bp)
- model_dir: the path of model directory
- out_fn: output file 

This script ouput the saliency map file for given sequences. The output could be visualized using saliency_heatmap.R. The visualization result is a pdf file which contains saliency map for each sequence.

```
python saliency_map.py -m model_type -in_fa input_fa -md model_dir -outfn out_fn
```

```R
Rscript saliency_heatmap.R saliency_map_out pdf_name
```
# Other

## 1. Confidence threshold

The table lists confidence thresholds of prediction scores for mammlian, human, mouse and zebrafish. Moderate, High and Very high correspond to 90%, 95% and 99% specificity respectively.

|           | Moderate | High  | Very high |
| --------- | -------- | ----- | --------- |
| Mammalian | 0.725    | 0.818 | 0.929     |
| Human     | 0.772    | 0.841 | 0.92      |
| Mouse     | 0.724    | 0.813 | 0.89      |
| Zebrafish | 0.715    | 0.82  | 0.9       |



