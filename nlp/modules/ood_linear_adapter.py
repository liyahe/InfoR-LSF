import logging
import torch
from sklearn.linear_model import LogisticRegression

from tools.utils import color


class OodLinearAdapter(torch.nn.Module):
    """Linear adapter for OOD evaluation, which is used to replace the original classifier."""

    def __init__(self, model, num_labels):
        super().__init__()
        self.ori_model = model
        self.num_labels = num_labels
        self.use_sgd = False
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)

    def forward(self, **inputs):
        outputs = self.ori_model(**inputs)
        logits = self.classifier(outputs["z"])
        if logits.shape[-1] == 1:  # binary classification, need to unsqueeze
            logits = torch.cat([-logits, logits], dim=-1)  # [N, 2]
        outputs["logits"] = logits
        return outputs

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor):
        logging.info(
            color(
                f"Linear adapter is training, training data shape: {embeddings.shape}, training data classes: {torch.bincount(labels)}",
                "blue",
            )
        )
        # fit a linear classifier
        if not self.use_sgd:
            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # Initialize and fit the Logistic Regression model
            classifier = LogisticRegression(random_state=0, max_iter=10000, multi_class="multinomial", solver="lbfgs")
            classifier.fit(embeddings_np, labels_np)

            # load the model as a torch module
            self.classifier.weight.data = torch.tensor(classifier.coef_, dtype=torch.float32)
            self.classifier.bias.data = torch.tensor(classifier.intercept_, dtype=torch.float32)
        else:
            # use SGD to train the linear classifier with pytorch, it seems that SGD is not as good as sklearn
            self.to(embeddings.device)
            self.classifier.weight.data.normal_(mean=0.0, std=0.01)
            self.classifier.bias.data.zero_()
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(100):
                optimizer.zero_grad()
                logits = self.classifier(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    logging.info(color(f"Linear adapter training epoch {epoch}, loss: {loss.item()}", "blue"))
        logging.info(
            color(
                f"Linear adapter is trained, weight shape: {self.classifier.weight.data.shape, self.classifier.bias.data.shape}",
                "blue",
            )
        )
