import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# check length for paper
metric_df = pd.read_csv("metric_df.csv")

print(metric_df.groupby("pair").tgt_.count())

target_score = ['mqm_score', 'comet_xl', 'comet_xxl', 'prometheus', 
       'gemba_all_orig', 'gemba_all_literary']
colx = ['mqm_score']
coly = ['prometheus', 'comet_xl', 'comet_xxl', 
       'gemba_all_orig', 'gemba_all_literary']

# corr
sns.set(font_scale = 1.2)
corr_df = pd.DataFrame()
for method in ["kendall"]:
    for p in ['de-en', 'en-de', 'de-zh', 'en-zh']: #,  'zh-en']:
        d1 = metric_df[metric_df.pair == p].fillna(0).groupby(["model_id"])[target_score].mean()
        corr = d1[target_score].corr(method = method).loc[colx, coly]
        corr["pair"] = p
        corr["method"] = method
        corr_df = pd.concat([corr_df, corr])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11,3))
sns.heatmap(corr_df.iloc[:, :-2], annot=True, fmt=".3f", ax = ax1, cmap = sns.color_palette("Blues", as_cmap=True), cbar=False)
ax1.set_yticks([0.5,1.5,2.5,3.5], ["De-En", "En-De", "De-Zh", "En-Zh"], rotation=0, size = 12)
ax1.set_xticks([0.5,1.5,2.5,3.5, 4.5], ["Prometheus 2", "XCOMET-XL", "XCOMET-XXL", "GEMBA-MQM (Original)", "GEMBA-MQM (Literary)"],
          rotation=45, size = 15)
ax1.set_title("(a)", size = 16)

target_score = ['mqm_score','gemba_acc_literary', 'gemba_flu_literary', "gemba_all_literary", 
       'gemba_sty_literary', 'gemba_term_literary', 'mqm_score_acc', 'mqm_score_fluency',
       'mqm_score_style', 'mqm_score_term',]
colx = ['mqm_score', 'mqm_score_acc', 'mqm_score_fluency',
       'mqm_score_style', 'mqm_score_term',]
coly = ['gemba_all_literary', 'gemba_acc_literary', 'gemba_flu_literary',
       'gemba_sty_literary', 'gemba_term_literary',]
# corr
tuple_ = [] 
corr_df = pd.DataFrame()
for method in ["kendall"]:
    for p in ['de-en', 'en-de', 'de-zh', 'en-zh']: #,  'zh-en']:
        d1 = metric_df[metric_df.pair == p].fillna(0).groupby(["model_id"])[target_score].mean()
        corr = d1[target_score].corr(method = method).loc[colx, coly]
        corr = tuple(np.diag(corr))
        tuple_.append(corr)
        
corr_df = pd.DataFrame(tuple_)
sns.heatmap(corr_df, annot=True, fmt=".3f", ax = ax2, cmap = sns.color_palette("Blues", as_cmap=True))
ax2.set_yticks([.5,1.5,2.5,3.5], ["De-En", "En-De", "De-Zh", "En-Zh"], rotation=0, size = 12)
ax2.set_xticks([0.5,1.5,2.5,3.5, 4.5], ["All", "Accuracy", "Fluency", "Style", "Terminology"], rotation=45, size = 15)
ax2.set_title("(b)", size = 16)

plt.savefig('corr_aspect.pdf',
           dpi=300,
           orientation='portrait', bbox_inches='tight', pad_inches=0)