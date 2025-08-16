import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
#from models.finch_matcher import build_matcher_crowd
from models.kmeans_matcher import build_matcher_crowd
#from models.matcher import build_matcher_crowd
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                    accuracy, get_world_size, interpolate,
                    is_dist_avail_and_initialized)

class ResNetBackbone(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {backbone}")
        
        # Remove the average pooling layer and fully connected layer
        self.base_layers = nn.ModuleList([
            nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool
            ),
            base_model.layer1,  # 256 channels for ResNet50, 64 for ResNet18/34
            base_model.layer2,  # 512 channels for ResNet50, 128 for ResNet18/34
            base_model.layer3,  # 1024 channels for ResNet50, 256 for ResNet18/34
            base_model.layer4   # 2048 channels for ResNet50, 512 for ResNet18/34
        ])
        
        self.backbone_type = backbone

    def forward(self, x):
        features = []
        for layer in self.base_layers:
            x = layer(x)
            features.append(x)
        return features
    
class DecoderResNet(nn.Module):
    def __init__(self, backbone='resnet50', feature_size=256):
        super(DecoderResNet, self).__init__()
        
        # Configure channel sizes based on backbone
        if backbone == 'resnet50':
            in_channels = [2048, 1024, 512]
        else:  # ResNet18/34
            in_channels = [512, 256, 128]
        
        self.P5_1 = nn.Conv2d(in_channels[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.P4_1 = nn.Conv2d(in_channels[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.P3_1 = nn.Conv2d(in_channels[2], feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.extra_down = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        C3, C4, C5 = inputs[-3:]
        
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        
        output = self.extra_down(P3_x)
        
        return output
    

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        # Check if x is 3D (batch_size, channels, pixels) and reshape if needed
        if len(x.shape) == 3:
            # Assuming x is [batch_size, channels, H*W]
            H = int(np.sqrt(x.shape[2]))
            x = x.view(x.shape[0], x.shape[1], H, H)
            
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)
        
        # Reshape the output appropriately
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        out = out.permute(0, 2, 1)
        
        return out.contiguous().view(batch_size, -1, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        # Check if x is 3D (batch_size, channels, pixels) and reshape if needed
        if len(x.shape) == 3:
            # Assuming x is [batch_size, channels, H*W]
            H = int(np.sqrt(x.shape[2]))
            x = x.view(x.shape[0], x.shape[1], H, H)
            
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)
        
        # Reshape the output appropriately
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        out = out.permute(0, 2, 1)
        out = out.contiguous().view(batch_size, -1, self.num_classes)
        
        return out



class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

def generate_anchor_points(stride=8, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points

def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

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
        #self.asyloss=ASLSingleLabel(gamma_neg=2, gamma_pos=0)
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


class P2PNetResNet(nn.Module):
    def __init__(self, backbone='resnet50', row=2, line=2):
        super().__init__()
        self.num_classes = 2
        num_anchor_points = row * line
        
        # Initialize ResNet backbone
        self.backbone = ResNetBackbone(backbone=backbone, pretrained=True)
        self.fpn = DecoderResNet(backbone=backbone)
        
        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256,
                                            num_classes=self.num_classes,
                                            num_anchor_points=num_anchor_points)
        
        self.anchor_points = AnchorPoints(pyramid_levels=[4], row=row, line=line)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        features = self.backbone(samples)
        features_fpn = self.fpn(features)
        
        batch_size = features_fpn.size()[0]
        
        regression = self.regression(features_fpn) * 100
        classification = self.classification(features_fpn)
        
        anchor_points = self.anchor_points(samples)
        
        regression = regression[:, :anchor_points.shape[1], :]
        classification = classification[:, :anchor_points.shape[1], :]
        
        output_coord = regression + anchor_points.repeat(batch_size, 1, 1)
        output_class = classification
        
        out = {'pred_logits': output_class, 'pred_points': output_coord}
        return out

def build(args, training=False):
    num_classes = 1
    backbone = args.get("backbone", "resnet50")  # Default to ResNet50 if not specified
    model = P2PNetResNet(backbone=backbone, row=args["row"], line=args["line"])
    
    if not training:
        return model
    
    weight_dict = {'loss_ce': args["loss_ce"], 'loss_points': args["loss_points"]}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes,
                                matcher=matcher,
                                weight_dict=weight_dict,
                                eos_coef=args["eos_coef"],
                                losses=losses)
    
    return model, criterion
