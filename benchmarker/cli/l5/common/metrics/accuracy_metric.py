import pytorch_lightning as pl


class Accuracy(pl.metrics.Accuracy):
    def update(self, preds, target):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.correct += sum(p == t for p, t in zip(preds, target))
        self.total += len(target)
