from yolo_backbone import*
import torch
import torch.nn as nn

class PairwiseGrindingYOLOClassifier(nn.Module):
    def __init__(self,
                 yolo_weights='yolov8n-cls.pt',
                 metadata_dim=2,
                 meta_hidden=16,
                 fusion_hidden=128,      # New: Fusion layer size
                 fc_hidden=64,
                 num_classes=2,
                 feature_layer_idx=-2,
                 dropout_p=0.5):
        super().__init__()
        # Shared image backbone for both images
        self.img_backbone = YOLOv8Backbone(
            weights=yolo_weights,
            feature_layer_idx=feature_layer_idx
        )
        # Shared metadata MLP for both metadata vectors
        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_dim, meta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(meta_hidden, meta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.zeros(1,3,640,640)
            feat = self.img_backbone(dummy)
            C_out = feat.shape[1]

        # Fusion layer: processes concatenated features before classifier
        self.fusion = nn.Sequential(
            nn.Linear(C_out + 2 * meta_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, img1, meta1, img2, meta2):
        """
        img1, img2:   [B, 3, 640, 640]
        meta1, meta2: [B, metadata_dim]
        """
        # Feature extraction
        feat1 = self.img_backbone(img1)
        feat2 = self.img_backbone(img2)
        meta_feat1 = self.meta_fc(meta1)
        meta_feat2 = self.meta_fc(meta2)

        # Concatenate features for each pair
        pair_feat = torch.cat([feat1 - feat2, meta_feat1, meta_feat2], dim=1)

        # Pass through fusion layer
        fused = self.fusion(pair_feat)

        # Classifier
        return self.classifier(fused)
    



def count_all_parameters(model):
    """Count ALL parameters in a model (trainable and non-trainable)"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """Count only trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create model instance
model = PairwiseGrindingYOLOClassifier(
    yolo_weights='yolov8n-cls.pt',
    metadata_dim=2,
    meta_hidden=16,
    fusion_hidden=128,
    fc_hidden=64,
    num_classes=2
)

# Print results
all_params = count_all_parameters(model)
trainable_params = count_trainable_parameters(model)

print(f"Total parameters: {all_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Detailed breakdown by component and layer
print("\nDetailed parameter breakdown:")
for name, module in model.named_modules():
    if name:  # Skip the empty-named top module
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        if num_params > 0:  # Only show layers with parameters
            print(f"  {name}: {num_params:,} parameters")
            
# Check if parameters are frozen in backbone
backbone_trainable = sum(p.requires_grad for p in model.img_backbone.parameters())
print(f"\nTrainable parameters in backbone: {backbone_trainable}/{len(list(model.img_backbone.parameters()))}")