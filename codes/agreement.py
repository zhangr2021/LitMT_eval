import ast
import pandas as pd
import numpy as np

# set the path to the downloaded dataset
path = ""
def gen_span_lst(df):
    entity_string = []
    for idx, row in df.iterrows():
        n = len(row["text"])
        init_label =["0"] * n
        if isinstance(row["label"], str):
            spans = ast.literal_eval(row["label"])
            for span in spans:
                init_label = [1 if (idx >= span["start"]) & (idx < span["end"]) else init_label[idx] for idx in range(len(init_label))]
        entity_string.append(init_label)
    return entity_string

all_ =[]
for pair in ["de-en", "en-de", "en-zh", "zh-en"]:
    df = pd.read_csv(path + "agreement/agreement_check - "+ pair + ".csv")
    df = df[["source", "annotator", "rating", "mqm_score", "model"]]
    df_ = pd.pivot(df,index=["source", 'model'],
                       columns=['annotator']).reset_index()
    #print(df_.columns, len(df))
    df_.columns = ["source_", "model", "rating1", "rating2", "mqm_score1", "mqm_score2"]
    corr = df_[["rating1", "rating2", "mqm_score1", "mqm_score2"]].corr(method = "kendall")
    all_.append((pair, "rating", corr.loc["rating1", "rating2"]))
    all_.append((pair, "mqm", corr.loc["mqm_score1", "mqm_score2"]))

corr = pd.DataFrame(all_, columns = ["pair", "mode", "corr"])
print(pd.pivot(corr,index=['mode'],
                       columns=['pair']).round(3))

from sklearn.metrics import cohen_kappa_score
import ast
def pair_wise_kappa(all_):
    span1, span2 = all_
    kappa_score = []
    for entity1, entity2 in zip(span1, span2):
        min_ = min(len(entity1), len(entity2))
        y1 = entity1[:min_]
        y2 = entity2[:min_]
        if (len(np.unique(entity1)) == 1) & (len(np.unique(entity2)) == 1):
            kappa_score.append(1)
        else:
            kappa_score.append(cohen_kappa_score(y1, y2))
    return kappa_score

tuple_lst = []
for pair in ["de-en", "en-de", "en-zh", "zh-en"]:
    all_ =[]
    df = pd.read_csv(path + "agreement/agreement_check - "+ pair + ".csv")
    df["source_"] = df.source.apply(lambda x: x[:20])
    df["tgt_"] = df.tgt.apply(lambda x: x[:20])
    df["pairs"] = df.groupby("tgt", as_index = False).model.transform("count")
    df = df[df["pairs"]>1].sort_values("tgt")
    print(len(df)/2)
    for id_, group in df.groupby("annotator"):
        group = group[["source_", "annotator", "rating", "mqm_score", "model", "label", "text"]]
        spans = gen_span_lst(group)
        all_.append(spans)
    kappa_score = pair_wise_kappa(all_)
    tuple_lst.append((pair, np.nanmean(kappa_score)))

print(pd.DataFrame(tuple_lst, columns = ["pair", "kappa"]).round(3))