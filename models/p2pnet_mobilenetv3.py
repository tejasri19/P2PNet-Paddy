import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized)

from models.backbone import build_backbone
from models.matcher import build_matcher_crowd
#from models.kmeans_matcher import build_matcher_crowd
from models.feature_extractor import SODModel
from  .ASLSingleLabel import ASLSingleLabel

torch.backends.cudnn.benchmark = True


class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Backbone, self).__init__()
        
        # Load pretrained MobileNetV3-Large
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            base_model = mobilenet_v3_large(weights=weights)
        else:
            base_model = mobilenet_v3_large()

        # Get feature layers from MobileNetV3
        features = list(base_model.features)
        
        # Split into stages for exact feature map extraction
        self.stage1 = nn.Sequential(*features[:6])    # -> 40 channels (C3)
        self.stage2 = nn.Sequential(*features[6:12])  # -> 112 channels (C4)
        self.stage3 = nn.Sequential(*features[12:])   # -> 960 channels
        
        # Project final features to required dimension
        self.proj = nn.Sequential(
            nn.Conv2d(960, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Extract features at three specific scales
        feat1 = self.stage1(x)      # C3 features
        feat2 = self.stage2(feat1)  # C4 features
        feat3 = self.stage3(feat2)  # High-level features
        feat3 = self.proj(feat3)    # Project to C5 features
        
        # Return exactly three feature maps
        return [feat1, feat2, feat3]  # [40, 112, 512] channels respectively


    
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=32):
        super(RegressionModel, self).__init__()
        
        self.num_anchor_points = num_anchor_points
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True)
        )

        # Output layer produces 2 coordinates per anchor point
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        # Debug prints
        #print(f"Regression input shape: {x.shape}")
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)
        
        #print(f"Pre-reshape regression shape: {out.shape}")
        
        batch_size, channels, height, width = out.shape
        
        # Reshape to (batch_size, height * width, num_anchor_points * 2)
        out = out.permute(0, 2, 3, 1).contiguous()
        num_predictions = height * width
        
        #print(f"Num predictions: {num_predictions}")
        #print(f"Num anchor points: {self.num_anchor_points}")
        
        # Reshape to have each prediction as a pair of coordinates
        reshaped = out.view(batch_size, num_predictions, self.num_anchor_points, 2)
        final_out = reshaped.view(batch_size, num_predictions * self.num_anchor_points, 2)
        
        #print(f"Final regression output shape: {final_out.shape}")
        
        return final_out

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, feature_size=32):
        super(ClassificationModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        #print(f"Classification input shape: {x.shape}")
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.output(out)
        
        #print(f"Pre-reshape classification shape: {out.shape}")
        
        batch_size, channels, height, width = out.shape
        out = out.permute(0, 2, 3, 1).contiguous()
        num_predictions = height * width
        
        # Reshape to get proper classification output
        reshaped = out.view(batch_size, num_predictions, self.num_anchor_points, self.num_classes)
        final_out = reshaped.view(batch_size, num_predictions * self.num_anchor_points, self.num_classes)
        
        #print(f"Final classification output shape: {final_out.shape}")
        
        return self.output_act(final_out)


# generate the reference points in grid layout
def generate_anchor_points(stride=8, row=1, line=1):
    """Generate anchor points in grid layout"""
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    """Shift anchor points to create a grid across the image"""
    # Create a grid of centers
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]  # number of anchor points
    K = shifts.shape[0]         # number of grid cells

    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((K, 1, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=2, line=2):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [2]  # Single level for matching dimensions
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [8]

        self.row = 1
        self.line = 1

    def forward(self, image):
        """Forward pass to generate anchor points"""
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        
        # Calculate output shape based on stride
        #feature_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        feature_shape = image_shape // 8  # This gives us 32x32 feature map
        
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)

        # Generate base anchor points (1 per cell)
        anchor_points = generate_anchor_points(
            stride=8,
            row=self.row,
            line=self.line
        )
        
        # Shift anchor points to create grid
        shifted_anchor_points = shift(
            shape=feature_shape,
            stride=8,
            anchor_points=anchor_points
        )
        
        all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)

        # Convert to tensor
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))



class Decoder(nn.Module):
    def __init__(self, C3_size=40, C4_size=112, C5_size=512, feature_size=256):
        super(Decoder, self).__init__()

        # Reduce channel dimensions and process C5 (Top-down path)
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # Process C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # Process C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1)
        self.P3_2 = nn.Conv2d(feature_size, 3, kernel_size=3, padding=1)  # Output 3 channels

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 3:
            raise ValueError(f"Expected 3 input feature maps, got {len(inputs) if isinstance(inputs, list) else 'non-list input'}")

        C3, C4, C5 = inputs

        # Top-down pathway
        P5_x = self.P5_1(C5)
        P5_x = self.relu(P5_x)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P5_x = self.relu(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = self.relu(P4_x)
        P4_x = P4_x + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P4_x = self.relu(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = self.relu(P3_x)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return P3_x  #
    
# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        self.num_anchor_points = 1
        
        # FPN/Decoder
        self.fpn = Decoder(
            C3_size=40,
            C4_size=112,
            C5_size=512,
            feature_size=256
        )
        
        # Number of anchor points should match between regression and classification
        self.regression = RegressionModel(
            num_features_in=3,
            num_anchor_points=self.num_anchor_points
        )
        
        self.classification = ClassificationModel(
            num_features_in=3,
            num_classes=self.num_classes,
            num_anchor_points=self.num_anchor_points
        )
        
        self.anchor_points = AnchorPoints(
            pyramid_levels=[2],
            row=row,
            line=line
        )

    def forward(self, samples: NestedTensor):
        if isinstance(samples, NestedTensor):
            x = samples.tensors
        else:
            x = samples
            
        #print(f"Input shape: {x.shape}")
        
        # Get backbone features and FPN output
        features = self.backbone(x)
        features_fpn = self.fpn(features)
        
        #print(f"FPN output shape: {features_fpn.shape}")
        
        # Get predictions
        regression = self.regression(features_fpn) * 100
        classification = self.classification(features_fpn)
        
        # Generate anchor points
        anchor_points = self.anchor_points(x)
        batch_size = x.shape[0]
        anchor_points = anchor_points.repeat(batch_size, 1, 1)
        
        #print(f"Regression shape: {regression.shape}")
        #print(f"Anchor points shape: {anchor_points.shape}")
        
        # Validate dimensions match
        if regression.shape[1] != anchor_points.shape[1]:
            raise ValueError(
                f"Dimension mismatch: regression {regression.shape} "
                f"vs anchor_points {anchor_points.shape}\n"
                f"Number of predictions: {regression.shape[1]}\n"
                f"Number of anchor points: {anchor_points.shape[1]}"
            )
            
        output_coord = regression + anchor_points
        output_class = classification
        
        return {'pred_logits': output_class, 'pred_points': output_coord}
    
    
    
class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.asyloss=ASLSingleLabel(gamma_neg=2, gamma_pos=0)
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        #loss_asyloss = self.asyloss(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}
        #losses = {'loss_ce': loss_asyloss}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_points'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)

        
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

    
    
# create the P2PNet model
def build(args, training):
    num_classes = 1
    backbone = MobileNetV3Backbone(pretrained=True)
    model = P2PNet(backbone, args["row"], args["line"])
    
    if not training:
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args["point_loss_coef"]}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes,
                                matcher=matcher,
                                weight_dict=weight_dict,
                                eos_coef=args["eos_coef"],
                                losses=losses)

    return model, criterion



