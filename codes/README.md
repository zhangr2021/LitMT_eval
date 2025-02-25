# Analysis Codes

This folder contains analysis scripts and data samples for results in the paper.

## Data Files
Note: Due to fair use restrictions, only truncated samples of the source and target texts (first 20 characters) are included. For full data access, please refer to the download link on the front page. 

- `bws_sampled_source.csv`: sampled source texts for bws.

- `metric_df.csv`: Contains metric scores and human evaluation results for model comparisons.

## Analysis Scripts

- `summary_statistics.py`: Calculates summary statistics for the evaluation results (dataset to be downloaded)

- `agreement.py`: Calculates annotation agreement among students (dataset to be downloaded)

- `metric_corr.py`: Computes and visualizes correlations between different metrics and human evaluations. Generates correlation heatmaps for:
  - Automatic metrics vs MQM scores
  - Aspect-specific correlations between GEMBA and human evaluations

- `ratio_compute.py`: Computes proportions of human translation that are preferred over MT results.

