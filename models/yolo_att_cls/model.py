import torch
import torch.nn as nn
from yolo_backbone import*

class AttentionFusionLayer(nn.Module):
    """
    Feature-wise attention fusion: metadata generates a gating vector
    for image features. Fused = img_feat * sigmoid(meta_attention)
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.feature_dim = feature_dim

    def forward(self, img_feat, meta_attn):
        # img_feat: [B, C]
        # meta_attn: [B, C] (already projected to same dim)
        attn = self.sigmoid(meta_attn)
        return img_feat * attn

class GrindingYOLOClassifier(nn.Module):
    def __init__(self,
                 yolo_weights='yolov8n-cls.pt',
                 metadata_dim=2,
                 meta_hidden=16,
                 fc_hidden=64,
                 num_classes=2,
                 feature_layer_idx=-2,
                 dropout_p=0.5):
        super().__init__()
        # 1) image feature extractor
        self.img_backbone = YOLOv8Backbone(
            weights=yolo_weights,
            feature_layer_idx=feature_layer_idx
        )
        # 2) metadata MLP projects to image feature dim
        with torch.no_grad():
            dummy = torch.zeros(1,3,640,640)
            feat = self.img_backbone(dummy)
            C_out = feat.shape[1]

        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_dim, meta_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(meta_hidden, C_out),  # Project to match image feature dim
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        # 3) attention fusion layer
        self.fusion = AttentionFusionLayer(feature_dim=C_out)
        # 4) final classifier
        self.classifier = nn.Sequential(
            nn.Linear(C_out, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, images, metas):
        img_feat  = self.img_backbone(images)
        meta_attn = self.meta_fc(metas)
        fused     = self.fusion(img_feat, meta_attn)
        return self.classifier(fused)