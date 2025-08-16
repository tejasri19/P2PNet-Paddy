import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import scipy.spatial

class HungarianMatcher(nn.Module):

    def __init__(self, args, cost_class: float = 1, cost_point: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.args = args
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"


    def k_distance(self,point_set1,point_set2,k=4):
        # Handle empty point sets
        if len(point_set1) == 0 or len(point_set2) == 0:
            return torch.zeros(len(point_set1), device=point_set1.device)
        distances = torch.cdist(point_set1, point_set2, p=1)
        # Adjust k if point_set2 has fewer points than k+1
        actual_k = min(k, len(point_set2) - 1)
        if actual_k <= 0:
            return torch.zeros(len(point_set1), device=point_set1.device)
        sorted_distances, _ = torch.sort(distances, dim=1)
        #print(sorted_distances)
        top_k_distances = sorted_distances[:, 1:k+1]
        mean_distances = torch.mean(top_k_distances, dim=1)
        return mean_distances


    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]


        out_point = outputs["pred_points"].flatten(0, 1)
        # Handle empty targets
        if not targets or all(len(v["point"]) == 0 for v in targets):
            # Return empty assignments for each batch
            return [(torch.tensor([], dtype=torch.int64), 
                    torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
            
        tgt_point = torch.cat([v["point"] for v in targets if len(v["point"]) > 0])
        tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"]) > 0])
        
        # If after filtering we have no points, return empty assignments
        if len(tgt_point) == 0:
            return [(torch.tensor([], dtype=torch.int64), 
                    torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        
        cost_point = torch.cdist(out_point, tgt_point.cuda(), p=1)
        
        
        k_distances = self.k_distance(tgt_point,tgt_point)
        cost_point = cost_point + k_distances
        #print("k_distances", k_distances)
        
        
        C = self.cost_class * cost_class + self.cost_point * cost_point
        #print("C", C)
        #C = C+k_distances
        #C = C+k_distances.T
        
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["point"]) for v in targets]
        # Filter out empty sizes
        valid_sizes = [s for s in sizes if s > 0]
        if not valid_sizes:
            return [(torch.tensor([], dtype=torch.int64), 
                    torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher_crowd(args):
    return HungarianMatcher(args, cost_class=args["set_cost_class"], cost_point=args["set_cost_point"])