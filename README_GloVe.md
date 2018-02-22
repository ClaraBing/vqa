* Pretrained GloVe vectors downloaded from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)

* Convert GloVe txt files to word2vec format:
```
# all glove*.txt are under /home/Bingbin/GloVe
>>> from gensim.scripts.glove2word2vec import glove2word2vec
>>> glove2word2vec(glove_input_file='/home/Bingbin/GloVe/glove.6B.300d.txt', word2vec_output_file='/home/Bingbin/GloVe/word2vec_glove.6B.300d.txt')
```

* Use word2vec in python
```
>>> from gensim.models.keyedvectors import KeyedVectors
>>> glove_model = KeyedVectors.load_word2vec_format("/home/Bingbin/GloVe/word2vec_glove.6B.300d.txt", binary=False)
```
