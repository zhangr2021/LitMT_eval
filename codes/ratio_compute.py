import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# check length for paper
metric_df = pd.read_csv("metric_df.csv")
bws = pd.read_csv("bws_sampled_source.csv")

for_ratio = metric_df.set_index("source_").loc[bws["source_"]]

def humanLLM(annotated):
    all_ = annotated.model_human.unique()
    all_result = pd.DataFrame()
    target_col = "model_human"
    for col in ["gemba_all_orig", "gemba_all_literary", "comet_xl", "comet_xxl", "mqm_score", "prometheus"]:
        print(col)
        for keys, group in annotated.groupby(["pair", "source_"]):
            p, s = keys
            set_ = set(group.model_human).intersection(set(all_))
            if "translator" in set_:
                t1 = [ 'deepl', 'gpt4o', 'google translate','qwen2-beta-7b-chat',]
                t1 = list(set(t1).intersection(set(set_)))
                t2 = ['qwen2-beta-7b-chat', 'm2m', 'nllb', 
                   'towerinstruct-7b-v0.2', 'meta-llama-3-8b-instruct', 'gemma-1.1-7b-it']
                t2 = list(set(t2).intersection(set(set_)))
                rank = len(group[col].rank()) - group[col].rank()+1
                group["rank"] = rank
                try:
                    r1 = group[[target_col, "rank"]].set_index(target_col).loc["translator"].min().values[0] 
                except: 
                    r1 = group[[target_col, "rank"]].set_index(target_col).loc["translator"].min()
                r2 = group[[target_col, "rank"]].set_index(target_col).loc[t1].min().values[0] 
                r3 = group[[target_col, "rank"]].set_index(target_col).loc[t2].min().values[0] 
                # better than all MT
                human_all_strict = r1 < r2 
                # better than or equal to all MT
                human_all = r1 <= r2
                # better than MT except GPT-4o, google translate, deepl
                human_Tmt_strict = r1 < r3
                human_Tmt = r1 <= r3
                tmp = pd.DataFrame((human_all_strict, human_all, human_Tmt_strict,  human_Tmt, p, s)).T
                tmp["mode"] = col
                all_result = pd.concat([all_result, tmp])  
                
    all_result.columns = ["human_all_strict", "human_all", "human_Tmt_strict", "human_mt", "pair", "source_", "mode"]
    all_result[["human_all_strict", "human_all", "human_Tmt_strict", "human_mt"]] = all_result[["human_all_strict", "human_all", "human_Tmt_strict", "human_mt"]].astype(bool)
    df_ = pd.pivot_table(all_result, index=['pair', 'source_'],
                       columns=['mode']).reset_index()
        
    return df_

df_ = humanLLM(for_ratio)
pd.options.display.float_format = '{:.2%}'.format
output = df_.groupby(["pair"])[[( 'human_all_strict', 'mqm_score'), ( 'human_all_strict', 'prometheus'), ( 'human_all_strict', 'comet_xl'), ( 'human_all_strict', 'comet_xxl'), ( 'human_all_strict', 'gemba_all_orig'),  ('human_all_strict',     'gemba_all_literary'), 
        ( 'human_Tmt_strict', 'mqm_score'),( 'human_Tmt_strict', 'prometheus'), ( 'human_Tmt_strict', 'comet_xl'), ( 'human_Tmt_strict', 'comet_xxl'), ( 'human_Tmt_strict', 'gemba_all_orig'), ( 'human_Tmt_strict', 'gemba_all_literary')]].mean().loc[["de-en", "en-de", "de-zh", "en-zh"]]
print(output)