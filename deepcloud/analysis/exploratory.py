import json
import os, re
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor

plt.rc('font', family='serif')

plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('legend', fontsize='x-large')

files = []
for file in os.listdir('data'):
    if re.match('^20170610.*jsons', file):
        files.append(file)

def parseJsons(file):
    with open('data/{}'.format(file)) as f:
        sl = []
        for i, line in enumerate(f):
            _s = pd.read_json(line, typ='series')
            sl.append(_s)
        df = pd.concat(sl, axis=1).T
        df['time'] = df['time'].astype(float)
        df['acc'] = df['acc'].astype(float)
        df['loss'] = df['loss'].astype(float)
    return df

for i, file in enumerate(files):
    if i == 0:
        df_all = parseJsons(file)
    else:
        df_all = df_all.append(parseJsons(file))

_dprint = df_all.groupby(['dataset', 'architecture','instance_type'])['time','loss'].aggregate({'time' : [np.mean, np.std], 'loss' : ['min']})
_dprint['Czas (s)'] = '$'+_dprint['time']['mean'].round(2).astype(str)+'+/-'+_dprint['time']['std'].round(2).astype(str)+'$'
_dprint.columns = _dprint.columns.get_level_values(0)

print(_dprint[['Czas (s)']].to_latex())

df_all = df_all.reset_index()
df_all['data_arch'] = df_all['dataset'] + '_' + df_all['architecture']

df_grp = df_all.sort_values(by='data_arch').groupby('data_arch')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[14, 18], sharey=False)
fig.subplots_adjust(hspace=0.35,wspace=0.1)
boxprops=dict(linewidth=1.5, color='1')

# Loop through each group and plot boxplot to appropriate axis
for i, k in enumerate(sorted(list(df_grp.groups.keys()))):
    group = df_grp.get_group(k)
    group.boxplot(ax=axes[floor(i/2)][i%2],
                  column='time',
                  by='instance_type',
                  boxprops=boxprops,
                  return_type='axes')
    axes[floor(i/2)][i%2].set_title('Zbiór danych: {}\nArchitektura: {}'.format(k.split('_')[0], k.split('_')[1]))
    axes[floor(i/2)][i%2].set_xlabel('Typ instancji')
    axes[floor(i/2)][i%2].set_ylabel('Czas obliczeń (s)')

fig.suptitle('')

fig.savefig('fig:experiment_results.png')
#plt.show()

_dprint = df_all.groupby(['dataset', 'architecture','instance_type'])['loss','val_loss'].aggregate({'acc' : ['max'], 'val_acc' : ['max'], 'loss' : ['min'], 'val_loss' : ['min']})
_dprint['Funkcja straty (zbiór treningowy)'] = '$'+_dprint['loss'].round(4).astype(str)+'$'
_dprint['Funkcja straty (zbiór walidacyjny)'] = '$'+_dprint['val_loss'].round(4).astype(str)+'$'
_dprint['Trafność (zbiór treningowy)'] = '$'+_dprint['acc'].round(4).astype(str)+'$'
_dprint['Trafność (zbiór walidacyjny)'] = '$'+_dprint['val_acc'].round(4).astype(str)+'$'
_dprint.columns = _dprint.columns.get_level_values(0)

print(_dprint.ix[:,4:9].to_latex())

get_ipython().magic('matplotlib inline')

tmp = df_all
tmp['timecum'] = df_all.sort_values(by='epoch').groupby(['data_arch','instance_type']).cumsum()['time']
df_grp = tmp.groupby('data_arch')

lsd = {'c4.2xlarge':'solid', 'p2.xlarge':'dashed'}

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[14, 18], sharey=False)
fig.subplots_adjust(hspace=0.35,wspace=0.25)

for i, k in enumerate(sorted(list(df_grp.groups.keys()))):
    group = df_grp.get_group(k)
    labels=[]
    for key, grp in group.groupby(['instance_type']):
        ax = grp.plot(ax=axes[floor(i/2)][i%2], kind='line', x='timecum', y='loss', c='black', ls=lsd.get(key))
        labels.append(key)
        lines, _ = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=1)
        if i == 0:
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(range(0, int(end)+1, 10000))
    axes[floor(i/2)][i%2].set_title('Zbiór danych: {}\nArchitektura: {}'.format(k.split('_')[0], k.split('_')[1]))
    axes[floor(i/2)][i%2].set_xlabel('Czas obliczeń (s)')
    axes[floor(i/2)][i%2].set_ylabel('Wartość funkcji celu')

fig.suptitle('')
fig.savefig('fig:experiment_loss.png')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[14, 18], sharey=False)
fig.subplots_adjust(hspace=0.35, wspace=0.25)

for i, k in enumerate(sorted(list(df_grp.groups.keys()))):
    group = df_grp.get_group(k)
    labels=[]
    for key, grp in group.groupby(['instance_type']):
        ax = grp.plot(ax=axes[floor(i/2)][i%2], kind='line', x='timecum', y='acc', c='black', ls=lsd.get(key))
        labels.append(key)
        lines, _ = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc=4)
        if i == 0:
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(range(0, int(end)+1, 10000))
    axes[floor(i/2)][i%2].set_title('Zbiór danych: {}\nArchitektura: {}'.format(k.split('_')[0], k.split('_')[1]))
    axes[floor(i/2)][i%2].set_xlabel('Czas obliczeń (s)')
    axes[floor(i/2)][i%2].set_ylabel('Trafność')

fig.suptitle('')
fig.savefig('fig:experiment_acc.png')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[14, 18], sharey=False)
fig.subplots_adjust(hspace=0.35,wspace=0.25)

for i, k in enumerate(sorted(list(df_grp.groups.keys()))):
    group = df_grp.get_group(k)
    for key, grp in group.groupby(['instance_type']):
        ax = grp.plot(ax=axes[floor(i/2)][i%2], kind='line', x='timecum', y='val_loss', c='black', ls=lsd.get(key))
        labels.append(key)
        if i == 0:
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(range(0, int(end)+1, 10000))
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    axes[floor(i/2)][i%2].set_title('Zbiór danych: {}\nArchitektura: {}'.format(k.split('_')[0], k.split('_')[1]))
    axes[floor(i/2)][i%2].set_xlabel('Czas obliczeń (s)')
    axes[floor(i/2)][i%2].set_ylabel('Wartość funkcji celu')

fig.suptitle('')
fig.savefig('fig:experiment_loss_val.png')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[14, 18], sharey=False)
fig.subplots_adjust(hspace=0.35,wspace=0.25)

for i, k in enumerate(sorted(list(df_grp.groups.keys()))):
    group = df_grp.get_group(k)
    for key, grp in group.groupby(['instance_type']):
        ax = grp.plot(ax=axes[floor(i/2)][i%2], kind='line', x='timecum', y='val_acc', c='black', ls=lsd.get(key))
        labels.append(key)
        if i == 0:
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(range(0, int(end)+1, 10000))
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    axes[floor(i/2)][i%2].set_title('Zbiór danych: {}\nArchitektura: {}'.format(k.split('_')[0], k.split('_')[1]))
    axes[floor(i/2)][i%2].set_xlabel('Czas obliczeń (s)')
    axes[floor(i/2)][i%2].set_ylabel('Trafność')

fig.suptitle('')
fig.savefig('fig:experiment_acc_val.png')
