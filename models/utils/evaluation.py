from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
import wandb
import json

def evaluate_and_log(results, output_file):
    ground_truth = [r["actual_constructs"] for r in results]
    predicted = [r["predicted_constructs"] for r in results]

    mlb = MultiLabelBinarizer()
    mlb.fit(ground_truth + predicted)
    Y_true = mlb.transform(ground_truth)
    Y_pred = mlb.transform(predicted)

    classification_metrics = classification_report(Y_true, Y_pred, output_dict=True, zero_division=0)
    avg_kappa = sum(cohen_kappa_score(Y_true[:, i], Y_pred[:, i]) for i in range(Y_true.shape[1])) / Y_true.shape[1]
    macro_f1 = f1_score(Y_true, Y_pred, average="macro")
    
    print(f"Classification Report:\n{classification_report(Y_true, Y_pred, target_names=mlb.classes_, zero_division=0)}")
    print(f"Average Cohen's Kappa across all labels: {avg_kappa:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    # wandb.init(project=project_name, name=run_name)
    # wandb.log({"classification_report": classification_metrics, "average_kappa": avg_kappa, "macro_f1": macro_f1})
    # wandb.save(output_file)


def f1_score(precision, recall):
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

def validate_construct(example, prediction, trace=None):
    """
    Compute Precision, Recall, and F1-Score for a single sample.
    """
    true_set = set(example.cfir_construct)
    pred_set = set(prediction.cfir_construct)
    
    # Precision: How many predicted constructs are correct
    precision = len(true_set & pred_set) / len(pred_set) if pred_set else 0.0
    
    # Recall: How many true constructs are predicted
    recall = len(true_set & pred_set) / len(true_set) if true_set else 0.0
    
    # F1-Score
    f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    
    return f1

   

    # wandb.finish()
