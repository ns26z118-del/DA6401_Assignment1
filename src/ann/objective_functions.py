import numpy as np

class MSELoss:
    def forward(self, y_pred, y_true):
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        if y_true.ndim == 1 and y_true.shape[0] != y_pred.shape[1]:
            # integer labels -> one-hot
            y_true = np.eye(y_pred.shape[1])[y_true.astype(int)]
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / (self.y_true.shape[0] * self.y_true.shape[1])


class CrossEntropyLoss:
    def forward(self, logits, y_true):
        # Ensure 2D
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Ensure y_true is integer class indices
        if np.ndim(y_true) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.y_true = np.array(y_true).flatten().astype(int)

        N = logits.shape[0]
        correct_logprobs = -np.log(self.probs[np.arange(N), self.y_true] + 1e-12)
        loss = np.mean(correct_logprobs)
        return loss

    def backward(self):
        N = self.y_true.shape[0]
        dZ = self.probs.copy()
        dZ[np.arange(N), self.y_true] -= 1
        dZ /= N
        return dZ