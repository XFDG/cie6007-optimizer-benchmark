"""Numerical manuscript fragments are generated from audited CSV files."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from src.utils import ROOT

NAMES=['sgd','adam','adamw','muon']
LABEL={'sgd':'SGD-M','adam':'Adam','adamw':'AdamW','muon':'Muon+AdamW'}


def main():
    s=pd.read_csv(ROOT/'results/summary.csv').set_index('optimizer')
    raw=pd.read_csv(ROOT/'results/metrics.csv')
    runs=pd.read_csv(ROOT/'results/run_summary.csv')
    chosen=json.loads((ROOT/'results/selected_config.json').read_text())['selected']
    tune=pd.read_csv(ROOT/'results/lr_sensitivity.csv')
    def value(n,col,scale=1,d=2):
        return f'{s.loc[n,col+"_mean"]*scale:.{d}f}'
    def pm(n,col,scale=1,d=2):
        return r'\pmv{'+value(n,col,scale,d)+'}{'+f'{s.loc[n,col+"_std"]*scale:.{d}f}'+'}'
    best=s.test_accuracy_mean.idxmax()
    fast=s.threshold_seconds_mean.idxmin()
    cheapest=s.total_train_seconds_mean.idxmin()
    budget=s.budget_test_accuracy_mean.idxmax()
    ratio=s.loc['muon','total_train_seconds_mean']/s.loc['adamw','total_train_seconds_mean']
    gap=(s.loc['muon','test_accuracy_mean']-s.loc['adamw','test_accuracy_mean'])*100
    abstract=rf'''\begin{{abstract}}
We compare momentum SGD, Adam, AdamW, and a Muon--AdamW hybrid on Fashion-MNIST using a fixed 335k-parameter MLP. Three learning rates per method are tuned on held-out training data; selected settings are evaluated with three paired seeds over 15 epochs. {LABEL[best]} attains the highest mean test accuracy, {pm(best,'test_accuracy',100)}\%, using validation-selected checkpoints. Muon uses {ratio:.2f}$\times$ AdamW's training time and differs in test accuracy by {gap:+.2f} percentage points. {LABEL[fast]} reaches 85\% validation accuracy fastest on average. Equal-step accuracy and time efficiency are reported separately, with a predeclared five-second trajectory readout. These measurements characterize one small MLP and one implementation; they do not establish a universal optimizer ranking.
\end{{abstract}}
'''
    (ROOT/'paper/generated_abstract.tex').write_text(abstract)
    code=(ROOT/'src/train.py').read_text()
    excerpt=code[code.index('    seed_all(seed)'):code.index('        row = dict(run_id=run_id')]
    (ROOT/'paper/generated_train_excerpt.py').write_text(excerpt)
    code=(ROOT/'scripts/evaluate_final.py').read_text()
    excerpt=code[code.index('    frozen = json.loads'):code.index('    pd.DataFrame(results)')]
    (ROOT/'paper/generated_eval_excerpt.py').write_text(excerpt)
    lines=[r'\paragraph{Accuracy and generalization (RQ2).}',
      'Table~\\ref{tab:accuracy} reports validation-selected checkpoints. '+
      f'{LABEL[best]} has the highest mean test accuracy ({pm(best,"test_accuracy",100)}\\%). '+
      'All values use three seeds, not repeated evaluations of one model. The final-epoch column checks whether checkpoint selection changes the interpretation.',
      r'\begin{table*}[t]\centering',
      r'\caption{Accuracy at validation-best checkpoints, with final-epoch test accuracy for comparison. Mean $\pm$ sample SD across three paired seeds; all accuracies and gaps are in percentage points. Muon+ denotes the hybrid.}',
      r'\label{tab:accuracy}',r'\begin{tabular}{lrrrrrr}\toprule',
      r'Method & Selected LR & Train & Validation & Test & Train--test gap & Final test\\\midrule']
    for n in NAMES:
        cells=[LABEL[n],f'{chosen[n]["lr"]:g}',pm(n,'train_accuracy',100),
               pm(n,'best_val_accuracy',100),pm(n,'test_accuracy',100),
               pm(n,'generalization_gap',100),pm(n,'final_test_accuracy',100)]
        lines.append(' & '.join(cells)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table*}',
      r'\begin{figure*}[t]\centering\includegraphics[width=.96\textwidth]{figures/convergence.pdf}',
      r'\caption{Training-set loss at epoch-end checkpoints and validation accuracy at equal epochs/updates. Lines are means; bands show one sample standard deviation over three seeds.}\label{fig:convergence}\end{figure*}',
      r'\paragraph{Convergence (RQ1).}',
      f'The first observed 85\\% validation crossing occurs at mean epochs '+
      ', '.join(f'{value(n,"threshold_epoch",d=2)} ({LABEL[n]})' for n in NAMES)+
      f'. {LABEL[fast]} is fastest by the corresponding training-plus-validation time. '+
      'Figure~\\ref{fig:convergence} shows that ranking by early progress need not match ranking by the final checkpoint. '+
      'Epoch-end training loss is evaluated on the full training subset, avoiding the changing-model average implicit in an online minibatch loss.',
      r'\paragraph{Computational efficiency (RQ3).}',
      f'{LABEL[cheapest]} has the lowest 15-epoch training cost. Muon takes {ratio:.2f}$\\times$ AdamW time '+
      f'({value("muon","total_train_seconds")} versus {value("adamw","total_train_seconds")} seconds). '+
      'The extra matrix operations and auxiliary optimizer calls are plausible causes of the overhead, but epoch timing alone does not attribute cost to individual kernels. '+
      'All methods have the same samples per epoch; throughput therefore expresses the same cost on a per-sample scale.',
      r'\begin{table*}[t]\centering',
      r'\caption{Measured efficiency. Train seconds exclude validation; threshold seconds include it. Throughput is in thousands of training samples/s. Peak allocated MiB includes GPU-resident development data.}',
      r'\label{tab:efficiency}\begin{tabular}{lrrrrr}\toprule',
      r'Method & Train (s) & Epoch (s) & k samples/s & Time to 85\% (s) & Peak (MiB)\\\midrule']
    for n in NAMES:
        lines.append(' & '.join([LABEL[n],pm(n,'total_train_seconds'),pm(n,'avg_epoch_seconds'),
            pm(n,'samples_per_second',.001),pm(n,'threshold_seconds'),pm(n,'peak_memory_mb',d=1)])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table*}',
      r'\begin{figure*}[t]\centering\includegraphics[width=.96\textwidth]{figures/efficiency_sensitivity.pdf}',
      r'\caption{Left: validation accuracy against measured cumulative training-plus-validation time; dotted line is the five-second budget. Right: validation-best accuracy for each of three learning-rate candidates (one tuning seed, ten epochs). Muon changes only its matrix learning rate; auxiliary AdamW stays at $10^{-3}$.}\label{fig:time}\end{figure*}',
      'At the five-second readout, validation-best-so-far checkpoints achieve test accuracies of '+
      ', '.join(f'{pm(n,"budget_test_accuracy",100)}\\% ({LABEL[n]})' for n in NAMES)+
      f'. {LABEL[budget]} has the highest mean under this specific accounted-time budget. '+
      'The budget selects different training durations; this is complementary to, rather than interchangeable with, the fixed-epoch comparison. A strict online deadline, data-loading cost, or a larger model could alter the ranking.',
      r'\paragraph{Learning-rate sensitivity (RQ4).}']
    spreads={n:100*(tune[tune.optimizer==n].best_val_accuracy.max()-
                         tune[tune.optimizer==n].best_val_accuracy.min()) for n in NAMES}
    lines.append('Across the three tested rates, the best-validation range spans '+
                 ', '.join(f'{spreads[n]:.2f} points ({LABEL[n]})' for n in NAMES)+
                 '. This range measures sensitivity within the specified grids, not over all rates. '+
                 'A single tuning seed can favor a configuration by chance; three equal trials do not guarantee equally comprehensive searches for different algorithms. '+
                 'Muon has an additional fixed auxiliary learning rate whose sensitivity is not explored. Its winning main rate, .005, lies at the lower grid boundary, so a better rate outside the grid remains possible.')
    lines += [r'\paragraph{Interpretation and limitations.}',
      f'Muon changes mean test accuracy relative to AdamW by {gap:+.2f} points while requiring {ratio:.2f}$\\times$ the training time. '+
      'Its approximate orthogonalization changes update geometry, but that mechanism alone does not imply a gain on this dataset. '+
      'Higher training accuracy is not sufficient evidence of better generalization: the train--test gaps in Table~\\ref{tab:accuracy} must be read with test accuracy, not minimized in isolation. '+
      'The gaps compare accuracies at the same checkpoint, not training accuracy averaged during optimization.',
      'The largest confounds are parameter routing (only 39.1\\% of parameters use Muon), fixed regularization strength, a short horizon, one dataset, and the small number of seeds. '+
      'A hybrid-control experiment with identical auxiliary AdamW for every method would help isolate the hidden-matrix optimizer. '+
      'The current experiment supports configuration-level conclusions only. '+
      'Muon reaches 94.03\\% mean training accuracy with a 4.99-point train--test gap, compared with AdamW at 92.68\\% and 4.09 points. The additional fitting transfers only partly to test accuracy. Adam and AdamW cross 85\\% validation at nearly the same measured time (0.99 versus 0.98 seconds), far smaller than their seed variation; this is not a meaningful speed separation. '+
      'Sample SD reflects seed variation on one split and GPU; it excludes uncertainty over data splits, hardware, implementation choices, and hyperparameter search. '+
      'No statistical significance or general superiority is claimed.']
    import re
    text='\n'.join(lines)+'\n'
    blocks=re.findall(r'\\begin\{(?:table|figure)\*\}.*?\\end\{(?:table|figure)\*\}',text,re.S)
    assert len(blocks)==4
    (ROOT/'paper/generated_accuracy_floats.tex').write_text('\n'.join(blocks[:2])+'\n')
    (ROOT/'paper/generated_efficiency_floats.tex').write_text('\n'.join(blocks[2:])+'\n')
    for b in blocks: text=text.replace(b,'')
    (ROOT/'paper/generated_results.tex').write_text(text)
    conclusion=(f'Under equal ten-epoch tuning trials and a 15-epoch final budget, {LABEL[best]} obtains the highest mean validation-selected test accuracy, '+
        f'while {LABEL[cheapest]} has the lowest training cost. '+
        f'Muon achieves {pm("muon","test_accuracy",100)}\\% test accuracy with an explicitly mixed parameter update scheme. '+
        'Accuracy per step, accuracy per second, and learning-rate robustness provide different evidence; no single metric justifies choosing an optimizer for every task. '+
        'Broader datasets, routing controls, and more tuning seeds would be the most useful next checks.\n')
    (ROOT/'paper/generated_conclusion.tex').write_text(conclusion)
    audit=json.loads((ROOT/'results/acceptance.json').read_text())
    (ROOT/'paper/generated_acceptance.tex').write_text(
      f'Thirteen optimizer/data/schema/resume unit tests passed. The experiment audit verified {audit["tuning_runs"]} tuning runs, '+
      f'{audit["formal_runs"]} formal runs, and {audit["epoch_rows"]} epoch rows including eight smoke rows. '+
      'It checks paired initial-weight hashes, equal tuning budgets, checkpoint hashes, validation-based selections, '+
      'and mean/sample-SD recomputation from raw data. Core files are available under '+
      '\\texttt{src/}, \\texttt{scripts/}, and \\texttt{tests/}. '+
      'Run \\texttt{python scripts/validate\\_results.py} after evaluation and plotting.\n')
    # Paired differences are saved without significance claims at n=3.
    paired=runs.pivot(index='seed',columns='optimizer',values='test_accuracy')
    out=[]
    for n in ['sgd','adam','muon']:
        delta=100*(paired[n]-paired['adamw'])
        out.append(dict(comparison=f'{n}-adamw',mean_pp=delta.mean(),std_pp=delta.std(),n=len(delta)))
    pd.DataFrame(out).to_csv(ROOT/'results/paired_differences.csv',index=False)

if __name__=='__main__': main()
