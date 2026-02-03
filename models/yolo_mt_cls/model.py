import torch
import torch.nn as nn
from models.yolo_pairs_cls.yolo_backbone import*

class GrindingYOLOMultiHeadClassifier(nn.Module):
    def __init__(
        self,
        yolo_weights='yolov8n-cls.pt',
        num_main_classes=None,
        num_grit_steps=None,
        num_grit_times=None,
        feature_layer_idx=-2
    ):
        super().__init__()

        print(num_main_classes)
        print(num_grit_steps)
        print(num_grit_times)

        # Defensive: Ensure all class counts are provided
        if num_main_classes is None:
            raise ValueError("num_main_classes must be specified")
        if num_grit_steps is None:
            raise ValueError("num_grit_steps must be specified")
        if num_grit_times is None:
            raise ValueError("num_grit_times must be specified")

        self.img_backbone = YOLOv8Backbone(
            weights=yolo_weights,
            feature_layer_idx=feature_layer_idx
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            feat = self.img_backbone(dummy)
            C_out = feat.shape[1]
            print('TASK', feat.shape[1])
            print('TASK', C_out)

        self.shared_head = nn.Sequential(
            nn.Linear(C_out, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.class_head = nn.Linear(128, num_main_classes)
        self.grit_step_head = nn.Linear(128, num_grit_steps)
        self.grit_time_head = nn.Linear(128, num_grit_times)

    def forward(self, images):
        img_feat = self.img_backbone(images)
        shared = self.shared_head(img_feat)

        class_logits = self.class_head(shared)
        grit_step_logits = self.grit_step_head(shared)
        grit_time_logits = self.grit_time_head(shared)

        return class_logits, grit_step_logits, grit_time_logits