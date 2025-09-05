import sentencepiece as spm
import sys
import json
import glob
import shutil

vocab_size = 120000
model_prefix = "base_sklarge_model/spiece"
traindir = "/mnt/sharedhome/hladek/bert-train/data/slavic1/sklarge-shards/"
srcfiles = glob.glob("train_sklarge_008*",root_dir=traindir)
print(srcfiles)

#traindir = "/home/jovyan/data/slavic-shards/"
#srcfiles = glob.glob( "train_slavic_173*",root_dir=traindir)

def line_generator():
    for fname in srcfiles:
        with open(traindir + "/" + fname) as f:
            for l in f:
                doc = json.loads(l)
                text = doc["text"]
                yield text


def line_generator3():
    for l in sys.stdin:
        doc = json.loads(l)
        text = doc["text"]
        yield text

def line_generator2():
    from datasets import load_dataset
    dataset = load_dataset("mc4","sk",streaming=True,split="validation")
    for doc in dataset:
        yield doc["text"]

spm.SentencePieceTrainer.train(
      sentence_iterator=line_generator(), 
      model_prefix=model_prefix, 
      shuffle_input_sentence=True,
      vocab_size=vocab_size,
      max_sentence_length=10000,
      train_extremely_large_corpus=True,
      pad_id=0,
      eos_id=1,
      unk_id=2,
      bos_id=3,
      )


