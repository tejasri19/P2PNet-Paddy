import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import scipy.spatial
from finch import FINCH

class HungarianMatcher(nn.Module):
    def __init__(self, args, cost_class: float = 1, cost_point: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.args = args
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    def finch_distance(self, point_set1, point_set2):
        """
        Compute distances using FINCH clustering instead of k-means
        """
        # Handle empty point sets
        if len(point_set1) == 0 or len(point_set2) == 0:
            return torch.zeros(len(point_set1), device=point_set1.device)
        
        # Convert to numpy for FINCH
        points = point_set2.cpu().numpy()
        
        # Run FINCH clustering
        partition, num_clusters, _ = FINCH(points, distance='euclidean', verbose=False)
        
        # Get the first partition level (most fine-grained clustering)
        clusters = partition[:, 0]
        
        # Compute cluster centers
        cluster_centers = []
        for i in range(num_clusters[0]):
            cluster_points = points[clusters == i]
            if len(cluster_points) > 0:
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
        
        cluster_centers = np.array(cluster_centers)
        
        # Convert back to torch tensor
        cluster_centers = torch.tensor(cluster_centers, device=point_set1.device)
        
        # Compute distances to cluster centers
        distances = torch.cdist(point_set1, cluster_centers.float(), p=1)
        
        # Get mean distance to nearest clusters
        min_distances, _ = torch.min(distances, dim=1)
        
        return min_distances

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_point = outputs["pred_points"].flatten(0, 1)

        # Handle empty targets
        if not targets or all(len(v["point"]) == 0 for v in targets):
            return [(torch.tensor([], dtype=torch.int64), 
                    torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
            
        tgt_point = torch.cat([v["point"] for v in targets if len(v["point"]) > 0])
        tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"]) > 0])
        
        # If after filtering we have no points, return empty assignments
        if len(tgt_point) == 0:
            return [(torch.tensor([], dtype=torch.int64), 
                    torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        # Compute classification cost
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute point matching cost with FINCH-based distances
        cost_point = torch.cdist(out_point, tgt_point.cuda(), p=1)
        finch_distances = self.finch_distance(tgt_point, tgt_point)
        cost_point = cost_point + finch_distances
        
        # Combine costs
        C = self.cost_class * cost_class + self.cost_point * cost_point
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["point"]) for v in targets]
        valid_sizes = [s for s in sizes if s > 0]
        
        if not valid_sizes:
            return [(torch.tensor([], dtype=torch.int64), 
                    torch.tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher_crowd(args):
    return HungarianMatcher(args, cost_class=args["set_cost_class"], cost_point=args["set_cost_point"])