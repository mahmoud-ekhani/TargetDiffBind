import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_sum
from torch_geometric.nn import knn_graph


class PropPredNet(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, output_dim=3):
        super(PropPredNet, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_channels
        self.output_dim = output_dim
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)

        self.encoder = get_encoder(config.encoder)
        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand,
                output_kind):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context_prop(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(
            node_attr=h_ctx,
            pos=pos_ctx,
            batch=batch_ctx,
        )  # (N_p+N_l, H)

        # Aggregate messages
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (N, F)
        output = self.out_block(pre_out)  # (N, C)
        if output_kind is not None:
            output_mask = F.one_hot(output_kind - 1, self.output_dim)
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)
        return output

    def get_loss(self, batch, pos_noise_std, return_pred=False):
        protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std
        ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std
        pred = self(
            protein_pos=batch.protein_pos + protein_noise,
            protein_atom_feature=batch.protein_atom_feature.float(),
            ligand_pos=batch.ligand_pos + ligand_noise,
            ligand_atom_feature=batch.ligand_atom_feature_full.float(),
            batch_protein=batch.protein_element_batch,
            batch_ligand=batch.ligand_element_batch,
            output_kind=batch.kind,
            # output_kind=None
        )
        # pred = pred * y_std + y_mean
        loss_func = nn.MSELoss()
        loss = loss_func(pred.view(-1), batch.y)
        if return_pred:
            return loss, pred
        else:
            return loss
        

def get_encoder(config):
    if config.name == 'egnn' or config.name == 'egnn_enc':
        net = EnEquiEncoder(
            num_layers=config.num_layers,
            edge_feat_dim=config.edge_dim,
            hidden_dim=config.hidden_dim,
            num_r_gaussian=config.num_r_gaussian,
            act_fn=config.act_fn,
            norm=config.norm,
            update_x=False,
            k=config.knn,
            cutoff=config.cutoff,
        )
    else:
        raise ValueError(config.name)
    return net


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    

def compose_context_prop(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    mask_protein = torch.cat([
        torch.ones([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.zeros([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx


class EnEquiEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian, k=32, cutoff=10.0,
                 update_x=True, act_fn='relu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian, fixed_offset=False)
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, node_attr, pos, batch):
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.net:
            h = h + interaction(h, edge_index, edge_attr)
        return h


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='relu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10. ** 2
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.r_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian, fixed_offset=False)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, edge_index, edge_attr):
        dst, src = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        mij = self.edge_mlp(torch.cat([edge_attr, hi, hj], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        # h = h + self.node_mlp(torch.cat([mi, h], -1))
        output = self.node_mlp(torch.cat([mi, h], -1))
        # if self.update_x:
        #     # x update in Eq(4)
        #     xi, xj = x[dst], x[src]
        #     delta_x = scatter_sum((xi - xj) * self.x_mlp(mij), dst, dim=0)
        #     x = x + delta_x

        return output
    

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class PropPredNetEnc(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim,
                 enc_ligand_dim, enc_node_dim, enc_graph_dim, enc_feature_type=None, output_dim=1):
        super(PropPredNetEnc, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_channels
        self.output_dim = output_dim
        self.enc_ligand_dim = enc_ligand_dim
        self.enc_node_dim = enc_node_dim
        self.enc_graph_dim = enc_graph_dim
        self.enc_feature_type = enc_feature_type

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + enc_ligand_dim, self.hidden_dim)
        # self.mean = target_mean
        # self.std = target_std
        # self.register_buffer('target_mean', target_mean)
        # self.register_buffer('target_std', target_std)
        self.encoder = get_encoder(config.encoder)
        if self.enc_node_dim > 0:
            self.enc_node_layer = nn.Sequential(
                nn.Linear(self.hidden_dim + self.enc_node_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim + self.enc_graph_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand,
                output_kind, enc_ligand_feature, enc_node_feature, enc_graph_feature):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        if enc_ligand_feature is not None:
            ligand_atom_feature = torch.cat([ligand_atom_feature, enc_ligand_feature], dim=-1)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx = compose_context_prop(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        h_ctx = self.encoder(
            node_attr=h_ctx,
            pos=pos_ctx,
            batch=batch_ctx,
        )  # (N_p+N_l, H)

        if enc_node_feature is not None:
            h_ctx = torch.cat([h_ctx, enc_node_feature], dim=-1)
            h_ctx = self.enc_node_layer(h_ctx)

        # Aggregate messages
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (N, F)
        if enc_graph_feature is not None:
            pre_out = torch.cat([pre_out, enc_graph_feature], dim=-1)

        output = self.out_block(pre_out)  # (N, C)
        if output_kind is not None:
            output_mask = F.one_hot(output_kind - 1, self.output_dim)
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)
        return output

    def get_loss(self, batch, pos_noise_std, return_pred=False):
        protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std
        ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std

        # add features
        enc_ligand_feature, enc_node_feature, enc_graph_feature = None, None, None
        if self.enc_feature_type == 'nll_all':
            enc_graph_feature = batch.nll_all  # [num_graphs, 22]
        elif self.enc_feature_type == 'nll':
            enc_graph_feature = batch.nll  # [num_graphs, 20]
        elif self.enc_feature_type == 'final_h':
            enc_node_feature = batch.final_h  # [num_pl_atoms, 128]
        elif self.enc_feature_type == 'pred_ligand_v':
            enc_ligand_feature = batch.pred_ligand_v  # [num_l_atoms, 13]
        elif self.enc_feature_type == 'pred_v_entropy_pre':
            enc_ligand_feature = batch.pred_v_entropy   # [num_l_atoms, 1]
        elif self.enc_feature_type == 'pred_v_entropy_post':
            enc_graph_feature = scatter(batch.pred_v_entropy, index=batch.ligand_element_batch, dim=0, reduce='sum')   # [num_graphs, 1]
        elif self.enc_feature_type == 'full':
            enc_graph_feature = torch.cat(
                [batch.nll_all, scatter(batch.pred_v_entropy, index=batch.ligand_element_batch, dim=0, reduce='sum')], dim=-1)
            enc_node_feature = batch.final_h
            enc_ligand_feature = torch.cat([batch.pred_ligand_v, batch.pred_v_entropy], -1)
        else:
            raise NotImplementedError

        pred = self(
            protein_pos=batch.protein_pos + protein_noise,
            protein_atom_feature=batch.protein_atom_feature.float(),
            ligand_pos=batch.ligand_pos + ligand_noise,
            ligand_atom_feature=batch.ligand_atom_feature_full.float(),
            batch_protein=batch.protein_element_batch,
            batch_ligand=batch.ligand_element_batch,
            output_kind=batch.kind,
            # output_kind=None,
            enc_ligand_feature=enc_ligand_feature,
            enc_node_feature=enc_node_feature,
            enc_graph_feature=enc_graph_feature
        )
        # pred = pred * y_std + y_mean
        loss_func = nn.MSELoss()
        loss = loss_func(pred.view(-1), batch.y)
        if return_pred:
            return loss, pred
        else:
            return loss