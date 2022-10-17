import torch
import speechbrain as sb


class SimpleBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        return self.modules.model(batch["input"])

    def compute_objectives(self, predictions, batch, stage):
        return torch.nn.functional.l1_loss(predictions, batch["target"])


model = torch.nn.Linear(in_features=10, out_features=10)
brain = SimpleBrain({"model": model}, opt_class=lambda x: torch.optim.SGD(x, 0.1))
data = [{"input": torch.rand(10, 10), "target": torch.rand(10, 10)}]
brain.fit(range(10), data)