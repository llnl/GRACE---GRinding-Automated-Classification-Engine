from ultralytics import YOLO
import torch
import torch.nn as nn
import warnings

class YOLOv8Backbone(nn.Module):
    def __init__(
        self,
        weights: str = 'yolov8n-cls.pt',
        feature_layer_idx: int = None
    ):
        """
        weights:            pretrained .pt path
        feature_layer_idx:  integer index in dm.model to STOP just before.
                            If None, stops at the Detect head.
                            Negative values are counted from the Detect head backwards.
        """
        super().__init__()
        # Load the nn.Moedule from the YOLO model
        dm = YOLO(weights).model
        # dm.model gives the layers as nn.Module(s)
        # I then make this into a nn.ModuleList (holds the "submodules")
        # Essentially, just extracts the layers from the YOLO model
        self.layers = nn.ModuleList(dm.model)

        # Returns segment will just return the index of the
        # first occurence of the "Detect" module in the list I made above
        # We want to replace this (or a layer a little before) with our binary classifier!
        self.detect_idx = next(i for i, m in enumerate(self.layers) if m.__class__.__name__ == "Classify")

        # Now, that we have found the detect layer, we want to
        # provide an option to choose where to slide RELATIVE
        # to the detect layer index.
        if feature_layer_idx is None:
            # Default is just the detection
            slice_idx = self.detect_idx
        elif feature_layer_idx < 0:
            # -2 -> two modules BEFORE Detect
            # Must add the plus 1 to INCLUDE the slice index
            slice_idx = self.detect_idx + feature_layer_idx + 1
        else:
            # If positive, extract up the the layer except NEVER
            # past the detect head.
            # This is a good work around for errors (I think?)
            slice_idx = min(feature_layer_idx, self.detect_idx)
        self.slice_idx = slice_idx

        # Print some setup logs for confirmation of behaviour
        print(
            f"Setup: Will run dynamic‐forward on modules [0:{slice_idx}]  "
            f"Setup(Detect is at {self.detect_idx})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the input image tensor
        # y is to store output of each layer (as the model runs)
        # out is the output of the current layer (which is then appeneded to y)
        y, out = [], x

        # Loop over each layer up to and including the slice
        for idx, m in enumerate(self.layers[: self.slice_idx]):
            # This is "from" -> a list of indices
            # If f = -1, means take the previous layer
            # Takes all of the output at the indices
            f = getattr(m, 'f', -1)
            # Get all the inputs to pass into the next layer
            if isinstance(f, (list, tuple)):
                inp = [out if j == -1 else y[j] for j in f]            
            elif f != -1:
                inp = y[f]
            else:
                inp = out

            # flatten single‐item lists so convs still get a Tensor
            if isinstance(inp, (list, tuple)) and len(inp) == 1:
                inp = inp[0]

            out = m(inp)   # module forward
            y.append(out)  # store for future skip connections

        # now `out` is your feature‐map at slice_idx−1
        return out.mean(dim=[2, 3])  # global average pool