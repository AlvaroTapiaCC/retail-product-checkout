from torch.utils.data import DataLoader

from ..paths import EMBEDDINGS_PATH, LAST_METRICS_PATH

from ..data.label_encoder import LabelEncoder
from ..data.dataset_loader import load_embeddings
from ..utils.io import save_metrics
from .metrics import (
    get_predictions,
    compute_all_metrics,
    plot_and_save_confusion_matrix,
)


def evaluate_mlp(model, val_path, device, batch_size):

    val_dataset = load_embeddings(val_path)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    label_encoder = LabelEncoder.load(
        EMBEDDINGS_PATH / "label_encoder.json"
    )

    y_true, y_pred = get_predictions(model, val_loader, device)

    metrics = compute_all_metrics(y_true, y_pred)

    save_metrics(metrics, LAST_METRICS_PATH / "metrics.json")

    plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        save_path=LAST_METRICS_PATH / "confusion_matrix.png",
        class_names=list(label_encoder.id2idx.keys())
    )

    return metrics