"""
This module define the abstract class Evaluator with its fundamental functions. The Abstract class is implemented by EvaluatorSota [[evaluator_sota.py]] and  EvaluatorLatency [[evaluator_sota.py]].

### Main Evaluation Functions

#### evaluate()
Evaluates the detector.
(See implementation: [[evaluator_sota.py#evaluate]], [[evaluator_latency.py#evaluate]])

#### eval_sota()
Evaluate detector based on state-of-the art metrics for classification

#### bin_preds_for_given_fpr()
Transform the prediciton probabilities output of the detector in binary predictions per FPR threshold

"""

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

    # === check_if_out_path_is_given ===
    
    """
    Utility method to determine the output path for saving results.
    
    Parameters:
    
    - results_p: Optional path provided as an argument.
    
    Returns:
    
    - The provided path if not None; otherwise, returns the default path stored in the object (`self.results_p`).
    
    Notes:
    
    - Ensures consistent behavior for saving files across methods without requiring path input every time.
    """
    def check_if_out_path_is_given(self, results_p):
        if results_p == None:
            results_p = self.results_p
        return results_p

    # === eval_sota ===

    """
    Computes standard evaluation metrics (SOTA) for binary classification and stores them.
     
    Parameters:
    
    - **test_y:** Ground truth binary labels for the test set.
    - **preds:** Binary predictions (0 or 1) made by the model.
    
    Returns:
    
    - **sota_results:** A dictionary or structured object containing accuracy, recall, precision, F1-score, False positive rate (FPR), and confusion matrix components (TN, FP, FN, TP).
    
    Notes:
    
    - Handles edge cases where the confusion matrix may have only one class present.
    - Computes FPR as `fp / (fp + tn)` and handles division errors.
    - Delegates result packaging to `store_sota_results()` in the `rh` utility module.
    """
    def eval_sota(self, test_y, preds):
        acc = sklearn.metrics.accuracy_score(test_y, preds)
        rec = sklearn.metrics.recall_score(test_y, preds)
        prec = sklearn.metrics.precision_score(test_y, preds)
        f1 = sklearn.metrics.f1_score(test_y, preds)
        cm = metrics.confusion_matrix(test_y, preds)
        
        if cm.shape == (1, 1):  
            tn, fp, fn, tp = (cm[0, 0], 0, 0, 0) if test_y[0] == 0 else (0, 0, 0, cm[0, 0])
        elif cm.shape == (2, 2): 
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

        try:
            fpr = fp / fp + tn
        except:
            fpr = None
        
        sota_results = rh.store_sota_results(acc, rec, prec, f1, fpr, tn, fp, fn, tp) 
        
        return sota_results

    # === bin_preds_for_given_fpr ===

    """
    Converts predicted probabilities into binary predictions based on specified false positive rates (FPRs).
    
    Parameters:
    
    - test_y: Ground truth binary labels for the test set.
    - preds_proba: Predicted class probabilities (2D array; positive class at index 1).
    - desired_fprs: List of target FPR thresholds to generate binary predictions for.
    - verbose: If True, prints the resulting binary predictions.
    
    Returns:
    
    - bin_preds_fpr: List of binary prediction arrays, one for each desired FPR.
    
    Notes:
    
    - Computes the ROC curve using `roc_curve()`.
    - For each desired FPR, finds the closest matching threshold and uses it to binarize predictions.
    - The result is a list of binary label arrays aligned with the requested FPR levels.
    """
    def bin_preds_for_given_fpr(self, test_y, preds_proba, desired_fprs, verbose=False):
        fpr, tpr, thresholds = metrics.roc_curve(test_y, preds_proba[:,1])
        fpr_indexes = [np.argmax(fpr > val) for val in desired_fprs] 
        fpr_thresholds = thresholds[fpr_indexes]
        bin_preds_fpr = [(preds_proba > val).astype(int)[:, 1] for val in fpr_thresholds]

        if verbose:
            print(bin_preds_fpr)
        
        return bin_preds_fpr
        