from abc import ABC, abstractmethod
from loader import PathManager
import sklearn
import results_handler as rh
from sklearn import metrics
import numpy as np

class Evaluator(ABC):
    
    def __init__(self, pm: PathManager):
        self.overall = {}
        self.results_p = pm.results_p

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Method to be implemented by subclasses"""
        pass

    def check_if_out_path_is_given(self, results_p):
        #if the path is not provided by argument take the one in object param.
        if results_p == None:
            results_p = self.results_p
        return results_p

    # === eval_sota ===
    def eval_sota(self, test_y, preds):
        
        """
        Computes classification metrics for evaluating an intrusion detection model.

        #### Args:
        ----------
        - `test_y (np.ndarray)`: Ground truth labels (0 for normal, 1 for anomalous).
        - `preds (np.ndarray)`: Predicted labels from the model (0 for normal, 1 for anomalous).

        #### Returns:
        ----------
        dict: A dictionary containing the following metrics:
            
        - `accuracy (float)`: Overall classification accuracy.
        - `recall (float)`: Proportion of actual positives correctly identified.
        - `precision (float)`: Proportion ofpositive predicted positivese.
        - `f1_score (float)`: Harmonic mean of precision and recall.
        - `fpr (float)`: False positive rate (FP / (FP + TN)).
        - `tn, fp, fn, tp (int)`: Confusion matrix values.
        """ 
        
        acc = sklearn.metrics.accuracy_score(test_y, preds)
        rec = sklearn.metrics.recall_score(test_y, preds)
        prec = sklearn.metrics.precision_score(test_y, preds)
        f1 = sklearn.metrics.f1_score(test_y, preds)
        cm = metrics.confusion_matrix(test_y, preds)
        
        if cm.shape == (1, 1):  # If there's only one class present
            tn, fp, fn, tp = (cm[0, 0], 0, 0, 0) if test_y[0] == 0 else (0, 0, 0, cm[0, 0])
        elif cm.shape == (2, 2):  # Normal case
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

        try:
            fpr = fp / fp + tn
        except:
            fpr = None
        
        sota_results = rh.store_sota_results(acc, rec, prec, f1, fpr, tn, fp, fn, tp) 
        
        return sota_results
        
    def bin_preds_for_given_fpr(self, test_y, preds_proba, desired_fprs, verbose=False):
        #compute the roc curve using model prediction probabilities 
        #select the index of the fpr to consider, find the thresholds for the desired FPRs, and compute binary predictions based on FPR thresholds 
        fpr, tpr, thresholds = metrics.roc_curve(test_y, preds_proba[:,1])
        fpr_indexes = [np.argmax(fpr > val) for val in desired_fprs] 
        fpr_thresholds = thresholds[fpr_indexes]
        bin_preds_fpr = [(preds_proba > val).astype(int)[:, 1] for val in fpr_thresholds]

        if verbose:
            print(bin_preds_fpr)
        
        return bin_preds_fpr
        