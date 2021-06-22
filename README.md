# Paper

## Using Document Embeddings for Background Linking of News Articles

Authors:

* Pavel Khloponin
* Leila Kosseim

https://link.springer.com/chapter/10.1007/978-3-030-80599-9_28

# Experiments data
https://docs.google.com/spreadsheets/d/1FIzv_2mvf47vfznNzeZqszYD4CNV4uGorAbC4rQ2ACA/edit?usp=sharing

# Initial Installation

```pip install -r requirements.txt```\
```python -m spacy download en_core_web_sm```

# Datasets
You have to request TREC Washington Post Corpus ( https://trec.nist.gov/data/wapost/ )
to use pipe1-wapo-spacy-preprocess.py

Alternatively you can use provided dataset based on Reuters news example ( https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz ) converted in format similar to WaPo dataset added to ./data/ directory

https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection
> II. Copyright & Notification 
>
>   The copyright for the text of newswire articles and Reuters
annotations in the Reuters-21578 collection resides with Reuters Ltd.
>Reuters Ltd. and Carnegie Group, Inc. have agreed to allow the free
>distribution of this data *for research purposes only*.  
>   If you publish results based on this data set, please acknowledge
>its use, refer to the data set by the name "Reuters-21578,
>Distribution 1.0", and inform your readers of the current location of
>the data set (see "Availability & Questions").


## for WaPo dataset
### 50 documents
```zcat ./data/TREC_Washington_Post_collection.v3.jl.gz |\ ```\
```head -50 |\ ```\
```./pipe1-wapo-spacy-preprocess.py - 2>&1 |\ ```\
```./pipe2-embedding.py > ./data/wapo.embedded.jl```
### all documents
```zcat ./data/TREC_Washington_Post_collection.v3.jl.gz |\ ```\
```./pipe1-wapo-spacy-preprocess.py - 2>&1 |\ ```\
```./pipe2-embedding.py > ./data/wapo.embedded.jl```


## for Reuters
### 50 documents
```zcat ./data/reuters-news.jl.gz |\ ```\
```head -50 |\ ```\
``` ./pipe2-embedding.py > ./data/wapo.embedded.jl```
### all documents
```zcat ./data/reuters-news.jl.gz | ./pipe2-embedding.py > ./data/wapo.embedded.jl```

# Normalisation
pipe3-normalyze.py expecting two blocks of data:
* first block of documents used to calculate statistics for embedding vectors
* used for vector normalisation on second block of document

These blocks are separated with line starting with \0 character (ASCII character with code 0)
Example:

```(cat ./data/wapo.embedded.jl; echo -e "\0"; cat ./data/wapo.embedded.jl ) |\ ```\
```./pipe3-normalize.py --size $(cat ./data/wapo.embedded.jl|wc -l) --type sigmoid```

It was implemented this way to be able to provide different datasets and apply scalers from first dataset on second one. 

# Usage

## ./pipe1-wapo-spacy-preprocess.py -h
```
usage: pipe1-wapo-spacy-preprocess.py [-h] bundle

Preprocess WashingtonPost docs to text

positional arguments:
  bundle      path to WaPo bundle to process: "./data/TREC_Washington_Post_collection.v3.jl.gz" or "-" to read from stdin

optional arguments:
  -h, --help  show this help message and exit

```

## ./pipe2-embedding.py -h
```
usage: pipe2-embedding.py [-h] [--text_field {text}] [--window_size {250,500,750,1000}] [--window_overlap {32,64,128}] [--batch_size BATCH_SIZE] [--device {['cuda', 'gpu', 'cpu', 'auto']}]
                          [--model {openai-gpt,transfo-xl-wt103,xlnet-base-cased,xlnet-large-cased,roberta-base,bert-base-uncased,bert-base-cased,bert-large-uncased,bert-large-cased,bert-base-multilingual-cased,bert-base-multilingual-uncased,google/pegasus-multi_news,google/pegasus-newsroom,gpt2,gpt2-medium,gpt2-large,gpt2-xl,xlnet-base-cased,xlnet-large-cased,roberta-base,roberta-large,distilroberta-base,roberta-base-openai-detector,roberta-large-openai-detector}]

converting text to embedding vector

optional arguments:
  -h, --help            show this help message and exit
  --text_field {text}   which field from json document should be used for embedding.
  --window_size {250,500,750,1000}
                        split document on chunks of this many tokens (feel free to extend)
  --window_overlap {32,64,128}
                        chunks should overlap with this many tokens (feel free to extend)
  --batch_size BATCH_SIZE
                        process this many documents in one batch
  --device {['cuda', 'gpu', 'cpu', 'auto']}
                        run torch on this device
  --model {openai-gpt,transfo-xl-wt103,xlnet-base-cased,xlnet-large-cased,roberta-base,bert-base-uncased,bert-base-cased,bert-large-uncased,bert-large-cased,bert-base-multilingual-cased,bert-base-multilingual-uncased,google/pegasus-multi_news,google/pegasus-newsroom,gpt2,gpt2-medium,gpt2-large,gpt2-xl,xlnet-base-cased,xlnet-large-cased,roberta-base,roberta-large,distilroberta-base,roberta-base-openai-detector,roberta-large-openai-detector}
                        which model to use for embedding

```

## ./pipe3-normalize.py -h
```
usage: pipe3-normalize.py [-h] [--repeat_delimiter REPEAT_DELIMITER] [--size SIZE] [--type {amplitude,sigmoid,none}]

Index WashingtonPost docs to ElasticSearch

optional arguments:
  -h, --help            show this help message and exit
  --repeat_delimiter REPEAT_DELIMITER
                        stream data this many loops
  --size SIZE           expected number of documents in first block
  --type {amplitude,sigmoid,none}
                        normalisation method to be used

```
