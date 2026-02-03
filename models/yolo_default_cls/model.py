import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, backbone, num_features, dropout_rate=0.5):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, 2)  # 2 outputs for binary

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits  # shape: (batch_size, 2)