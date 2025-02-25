import pandas as pd
import numpy as np
import nltk.data
import re
#nltk.download('punkt')

# load dataset: replace with the path to the dataset
d1 = pd.read_csv("human evalution/Student_annotation_2_de-en.csv", sep = ";")
d2 = pd.read_csv("human evalution/Student_annotation_3_en-de.csv", sep = ";")
d3 = pd.read_csv("human evalution/Student_annotation_4_de-zh.csv", sep = ";")
d4 = pd.read_csv("human evalution/Student_annotation_5_en-zh.csv", sep = ";")
d5 = pd.read_csv("agreement/Student_annotation_1_en-zh.csv")
d6 = pd.read_csv("agreement/Student_annotation_2_en-de.csv")
d7 = pd.read_csv("agreement/Student_annotation_3_de-en.csv")
annotated = pd.concat([d1,d2,d3,d4, d5, d6, d7])

# calculate length for texts    
def calculate_length(text, pair, return_1 = True, mode = "src"):
    src_lang, tgt_lang = pair.split("-")
    if mode == "src":
        lang = src_lang
    else:
        lang = tgt_lang
    if (lang == "en") or (lang == "de"):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        length = len(text.split())
        num_sent = len(tokenizer.tokenize(text))
    else:
        length = len(text.replace("\n", ""))
        num_sent = max(len(re.findall('[.？?。]', text)),1)
    if return_1:
        return length
    else:
        return num_sent

annotated["source"] = annotated["source"].apply(lambda x: x.replace("\n ", "\n").replace("\n\n\n", "\n\n"))
annotated["source_"] = annotated["source"].apply(lambda x: x[:20]) # first 20 words for src matching    
print("Number of unique source paragraphs: ", annotated.groupby("pair").source_.nunique())
print("Number of annotated paragraphs for each pair: ", annotated.groupby("pair").count().iloc[:,1])

#  calculate length for texts
annotated["length_src"] = annotated.apply(lambda x: calculate_length(x["source"], x["pair"],True), axis = 1)
annotated["nsent_src"] = annotated.apply(lambda x: calculate_length(x["source"], x["pair"], False), axis = 1)
annotated["token_sent_src"] = annotated["length_src"]/annotated["nsent_src"] 

annotated["length_tgt"] = annotated.apply(lambda x: calculate_length(x["tgt"], x["pair"], True,mode = "tgt"), axis = 1)
annotated["nsent_tgt"] = annotated.apply(lambda x: calculate_length(x["tgt"], x["pair"], False, mode = "tgt"), axis = 1)
annotated["token_sent_tgt"] = annotated["length_tgt"]/annotated["nsent_tgt"] 

print("Average length of source for each pair: ", annotated[["source", "length_src", "nsent_src", "token_sent_src", "pair"]].drop_duplicates().groupby("pair")[["length_src", "nsent_src", "token_sent_src"]].mean().loc[['de-en', 'en-de','de-zh', 'en-zh']].round(1))
print("Average length of source for all pairs: ", annotated[["source", "length_src", "nsent_src", "token_sent_src", "pair"]].drop_duplicates()[["length_src", "nsent_src", "token_sent_src"]].mean().round(1))

print("Average length of target for each pair: ", annotated.groupby("pair", as_index=False)[["length_tgt", "nsent_tgt", "token_sent_tgt"]].mean().set_index("pair").loc[['de-en', 'en-de','de-zh', 'en-zh',]].round(1)) 
print("Average length of target for all pairs: ", annotated[["length_tgt", "nsent_tgt", "token_sent_tgt"]].mean().round(1))

