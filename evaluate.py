#script performing evaluations for synthetic datasets

from evaluate_util import data_loader,Fidelity
from report_util import Fidelity
from preprocess import DGAN_pr

#load mock dataset (or real & synthetic dataset in the future) and preprocess to 3d array
attr = data_loader('data/mock_attr.csv')
long = data_loader('data/mock_longitudinal.csv')
attr,long = DGAN_pr.preprocess_mock(attr,long)

#create dictionary of descriptive statistics as first sanity check
attr_stats,long_stats = Fidelity.descr_stats(attr=attr,long=long)

#create desciptive statistic table
#Fidelity.stats_table(long_stats,statistics=['mean','stdev'],features=['feat1','feat2','feat3','feat4','feat5']).show()


# #create percentile plots
percentiles = [5,25,50,75,95]
alphas= [.2,.5,1,.5,.2]
Fidelity.percentile_plot(long_stats,percentiles,alphas).show()

# #create tsne plots
# tsne_embeddings = Fidelity.tsne(long)
# Fidelity.tsne_plot(tsne_embeddings).show()

#create gof report
# dct = Fidelity.gof(meta_real=meta[0:50],long_real=long[0:50],meta_syn=meta[50:],long_syn=long[50:])s
# rep = Fidelity.gof_report(dct=dct)
# print(rep)