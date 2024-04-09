"""
Created April 9, 2024

@author: Nate Gaylinn

Hyperparameter sweep for the user preferences model.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc

import evaluate

# This seems to suggest that the l2 penalty results in better roc_auc scores
# and that the ideal regularization coefficient is C ~= 1.129
def main():
    combined_df = evaluate.user_independent_user_pref()
    print(f'            | penalty = l1    | penalty = l2    |')
    print(f'-------------------------------------------------')
    # Try a range of different degrees of regularization. The default is 1.0,
    # and smaller values indicate MORE regularization. So, test a range of
    # values from 0.01 to 10.0.
    for C in np.logspace(-2, 1, num=20):
        print(f' {C = :<7.3f}|', end='')
        # Liblinear is much more efficient on this dataset than the other
        # solvers, and produces similar results. It accepts either l1 or l2
        # norm as a penalty.
        for penalty in ['l1', 'l2']:
            kwargs = { 'penalty':penalty, 'C':C, 'solver':'liblinear' }
            accuracies, f1s, true_labels, _, pred_scores = (
                evaluate.k_fold_CV(combined_df, kwargs))
            fpr, tpr, _ = roc_curve(true_labels, pred_scores)
            roc_auc = auc(fpr, tpr)
            # We're optimizing for AUC, but could also use accuracy or f1
            # score:
            # avg_accuracy = np.mean(accuracies)
            # avg_f1 = np.mean(f1s)
            print(f' {roc_auc = :<5.3f} |', end='')
        print()



if __name__ == '__main__':
    main()
