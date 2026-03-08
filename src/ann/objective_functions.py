

import numpy as np

class MSELoss:
    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):

        # N = self.y_true.shape[0]
        # return 2 * (self.y_pred - self.y_true) / N
        return 2 * (self.y_pred - self.y_true) / (self.y_true.shape[0] * self.y_true.shape[1])


class CrossEntropyLoss:
    def forward(self, logits, y_true):

        logits = logits - np.max(logits, axis=1, keepdims=True)

        exp_scores = np.exp(logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.y_true = y_true
        N = logits.shape[0]

        correct_logprobs = -np.log(self.probs[np.arange(N), y_true])
        loss = np.mean(correct_logprobs)
        return loss

    def backward(self):

        N = self.y_true.shape[0]
        dZ = self.probs.copy()
        dZ[np.arange(N), self.y_true] -= 1
        dZ /= N
        return dZ
    

