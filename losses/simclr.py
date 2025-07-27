import torch
import torch.nn.functional as F
from utils.dist_utils import gather_tensor, get_rank

def simclr_loss_distributed(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    gathered_z = gather_tensor(z)
    ic(z.size(), gathered_z.size())
    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)
    ic(sim.size())
    gathered_indexes = gather_tensor(indexes)
    ic(gathered_indexes.size())
    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    ic(pos_mask.size())
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    ic(z.size(0) * get_rank())
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    ic(pos, neg)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss

def simclr_loss(z: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Computes SimCLR loss given a batch of projected features.

    Args:
        z (torch.Tensor): (2N, D) Tensor containing projected features from two augmented views of a batch.
        temperature (float): Temperature parameter for scaling the logits.

    Returns:
        torch.Tensor: SimCLR loss.
    """
    # Normalize the feature vectors
    z = F.normalize(z, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(z, z.T)
    ic(similarity_matrix)

    # Mask to exclude self-similarities
    batch_size = z.size(0) // 2

    # labels = torch.cat([torch.arange(batch_size//2) for i in range(batch_size//2)], dim=0)
    # ic(labels)
    # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # ic(labels)
    # labels = labels.to(z.device)
    # ic(labels.shape)

    mask = torch.eye(batch_size*2, dtype=torch.bool).to(z.device)
    ic(mask.size())
    # labels = labels[~mask].view(labels.size(0), -1)
    # similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)
    ic(mask)
    # Compute the loss

    # positives = similarity_matrix[labels.bool()].view(labels.size(0), -1)
    # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.size(0), -1)
    positives = torch.diag(similarity_matrix, batch_size)
    positives = torch.cat([positives, torch.diag(similarity_matrix, -batch_size)])

    ic(positives)

    mask = (~torch.eye(batch_size*2, dtype=bool)).to(z.device)
    ic(mask)
    similarity_matrix = similarity_matrix[mask].view(batch_size*2, -1)
    ic(similarity_matrix.shape)
    
    logits = similarity_matrix / temperature
    labels = torch.arange(batch_size).repeat(2).to(z.device)

    ic(logits.shape, labels.shape)
    ic(logits, labels)
    loss = F.cross_entropy(logits, labels)
    ic(loss, loss.min(), loss.max())
    return loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size=2, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).cuda()).float(),
        )

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature
        )  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
