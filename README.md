# Elote


## Summary
Elote is a Python tool based in Transformers focused on analyzin the gender bias for translations made my the **[MarianMT](https://huggingface.co/transformers/model_doc/marian.html)** Framework which is being developed by the Microsoft Translator Team. Marian MT Machine translation models trained using [OPUS](http://opus.nlpl.eu/) data by Jörg Tiedemann

Running the project is as simple as go to terminal, do not forget to ensure all the dependencies are installed.
After that, just switch the path to the project folder(Elote) and run the following command to execute the main.py file:
    
```sh
$ python3 main.py
```

## Project structure 
  - main.py - Root file of the project.
  - Marian_evaluator.py - This file classifies, translates, and separates the sentences to analyze.    
  - twt.train.txt -  Corpus of several tweets in English.
  - candidate_sentences.txt - A file contained a list of sentences using the article "the".
  - candidate_sentences_translated.txt -  A file contained a list of translated sentences by the MArian MT Framework. 
  - results_multi_gender_sentences.txt -  A file contained a list of multi gender sentences to me manually analyzed.

  

## Dependencies
The following dependencies are required to run Elote:
- [Anaconda ](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html)
- [Pytorch  ](https://pytorch.org/get-started/locally/)
- [Torchvision ](http://pytorch.org/vision/stable/index.html)
- [SentencePiece ](https://github.com/google/sentencepiece#installation)  


