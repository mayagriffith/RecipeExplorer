"""
Created April 9, 2024

@author: Nate Gaylinn, Alaina Birney

Hyperparameter sweep for the user preferences model.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc

import evaluate

# This seems to suggest that the l2 penalty results in better roc_auc scores
# and that the ideal regularization coefficient is C ~= 1.129
def main():
    reg_or_svm = input("Are you evaluating the logistic regression or svm user preference model? (enter reg or svm)")
    combined_df = evaluate.user_independent_user_pref()
    if reg_or_svm == "reg":
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
                    evaluate.k_fold_CV(combined_df, reg_or_svm, log_reg_args = kwargs, svm_args=None))
                fpr, tpr, _ = roc_curve(true_labels, pred_scores)
                roc_auc = auc(fpr, tpr)
                # We're optimizing for AUC, but could also use accuracy or f1
                # score:
                # avg_accuracy = np.mean(accuracies)
                # avg_f1 = np.mean(f1s)
                print(f' {roc_auc = :<5.3f} |', end='')
            print()
    if reg_or_svm == "svm":
        results = []
        # try different C, kernels, and gamma
        print('            |                 | kernel = rbf   | kernel = poly | kernel = sigmoid')
        print('---------------------------------------------------------------------------------')
        # testing C from .001 (more comprimise) to 100 (less comprimise)
        for C in np.logspace(-3,2,num=15):
            for gamma in np.logspace(-3,2,num=15):
                print(f' {C = :<7.3f}|', end='')
                print(f" {gamma = :<7.3f}|", end="")
                for kernel in ["rbf", "poly", "sigmoid"]:
                    kwargs = {"kernel": kernel, "C": C, "probability": True, "gamma": gamma}
                    accuracies, f1s, true_labels, _, pred_scores = (
                        evaluate.k_fold_CV(combined_df, reg_or_svm, log_reg_args = None, svm_args = kwargs))
                    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
                    roc_auc = auc(fpr, tpr)
                    results.append({"C": C, "gamma": gamma, "kernel": kernel, "roc_auc": roc_auc})
                    print(f' {roc_auc = :<5.3f} |', end='')
                print()
        best_result = max(results, key=lambda x: x["roc_auc"])
        print("The following configuration was found to result in the highest ROC AUC within the range of values tested:")
        print(best_result)

if __name__ == '__main__':
    main()
