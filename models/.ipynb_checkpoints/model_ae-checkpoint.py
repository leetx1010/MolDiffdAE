from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch_geometric.data import Batch
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet

from .common import *
from .diffusion import *

class MolEncoder(Module):
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,  # explicite bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_timesteps = 0

        # # embedding
        node_dim = config.encoder.node_dim
        edge_dim = config.encoder.edge_dim
        emb_dim = config.encoder.emb_dim
        time_dim = config.diff.time_dim if self.num_timesteps > 0 else 0
        self.node_embedder = nn.Linear(num_node_types, node_dim-time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim-time_dim, bias=False) # the init edge features
        if self.num_timesteps != 0:
            self.time_emb = GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear')
        # # predictor
        self.encoder = NodeEdgeNet(**config.encoder)
        self.edge_decoder = MLP(edge_dim + node_dim, num_edge_types, edge_dim, num_layer=3)
        
        self.edge_weight = torch.tensor([0.1]+[1.]*(self.num_edge_types-1), dtype=torch.float32)
        self.ce_loss = torch.nn.CrossEntropyLoss(self.edge_weight)

        self.final = MLP(node_dim + edge_dim, emb_dim, emb_dim*2, num_layer=2)


    def forward(self, h_node, pos_node, batch_node, h_edge, edge_index, batch_edge):
        """
        Predict the edge type of edges defined by halfedge_index
        """
        
        # embedding 

        if self.num_timesteps != 0:
            time_embed_node = self.time_emb(t.index_select(0, batch_node))
            h_node = torch.cat([self.node_embedder(h_node), time_embed_node], dim=-1)
            time_embed_node = self.time_emb(t.index_select(0, batch_edge))
            h_edge = torch.cat([self.edge_embedder(h_edge), time_embed_node], dim=-1)
        else:
            h_node = self.node_embedder(h_node)
            h_edge = self.edge_embedder(h_edge)
            t = torch.zeros(batch_node.max()+1, device=pos_node.device, dtype=torch.long)

        h_node, _, h_edge = self.encoder(
            h_node=h_node,
            pos_node=pos_node, 
            h_edge=h_edge, 
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / max(self.num_timesteps, 1),
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / max(self.num_timesteps, 1),
        )

        h_sub = []

        for k in range(batch_node.max()+1):
            h_node_sub = h_node[batch_node==k].mean(0)
            h_edge_sub = h_edge[batch_edge==k].mean(0)
        
            h_sub.append(torch.cat([h_node_sub, h_edge_sub], dim=0))

        h_sub = torch.vstack(h_sub)
        emb = self.final(h_sub)

        
        return emb, batch_node



class MolDiffAE(Module):
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        emb_dim = config.encoder.emb_dim

        self.emb_dim = emb_dim
        self.wass_weight = config.encoder.wass_weight

        node_input_dim = node_dim+emb_dim+time_dim
        edge_input_dim = edge_dim+emb_dim+time_dim
        
        
        self.node_embedder = nn.Linear(num_node_types, node_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )

        # # encoder
        self.encoder = MolEncoder(config=config,
                                  num_node_types=num_node_types,
                                  num_edge_types=num_edge_types)
        
        # # denoiser
        if config.denoiser.backbone == 'NodeEdgeNet':
            self.denoiser = NodeEdgeNet(node_input_dim, edge_input_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.node_decoder = MLP(node_input_dim, num_node_types, node_dim)
        self.edge_decoder = MLP(edge_input_dim, num_edge_types, edge_dim)



    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 4., 8.]) # for earlier models: [1,4,8] default; [1,1,1] for noscale
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.num_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        # sample time
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def add_noise(self, node_type, node_pos, batch_node,
                    halfedge_type, halfedge_index, batch_halfedge,
                    num_mol, t, bond_predictor=None, **kwargs):
            num_graphs = num_mol
            device = node_pos.device

            time_step = t * torch.ones(num_graphs, device=device).long()

            # 2.1 perturb pos, node, edge
            pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
            node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
            halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
            # edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
            # batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
            if self.categorical_space == 'discrete':
                h_node_pert, log_node_t, log_node_0 = node_pert
                h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
            else:
                h_node_pert, h_node_0 = node_pert
                h_halfedge_pert, h_halfedge_0 = halfedge_pert
            return [h_node_pert, pos_pert, h_halfedge_pert]

    def get_loss(self, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge,
                num_mol
    ):
        num_graphs = num_mol
        device = node_pos.device

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

       
        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert

            h_node = F.one_hot(node_type, self.num_node_types).float()
            h_halfedge = F.one_hot(halfedge_type, self.num_edge_types).float()
            h_edge = torch.cat([h_halfedge, h_halfedge], dim=0)
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert

            h_node = h_node_0
            h_edge = torch.cat([h_halfedge_0, h_halfedge_0], dim=0)
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # 2.2 embedding
        emb = self.encoder(h_node, node_pos, batch_node, h_edge, edge_index, batch_edge)[0]


        # 3. forward to denoise
        preds = self(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, 
            time_step,
            emb
        )
        pred_node = preds['pred_node']
        pred_pos = preds['pred_pos']
        pred_halfedge = preds['pred_halfedge']

        # 4. loss
        # 4.1 pos
        loss_pos = F.mse_loss(pred_pos, node_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(node_pos[bond_index[0]] - node_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_pos[bond_index[0]] - pred_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)
    
        if self.categorical_space == 'discrete':
            # 4.2 node type
            log_node_recon = F.log_softmax(pred_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, batch_node, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, batch_node, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=batch_node)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type
            log_halfedge_recon = F.log_softmax(pred_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, batch_halfedge, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, batch_halfedge, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, 
                            log_edge_post_pred, log_halfedge_0, t=time_step, batch=batch_halfedge)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_node, h_node_0)  * 30
            loss_edge = F.mse_loss(pred_halfedge, h_halfedge_0) * 30

        loss_wass = kernel_mmd_loss(emb)*self.wass_weight

        # total
        loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0) + (loss_wass if self.wass_weight != 0.0 else 0)
        
        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'loss_wass': loss_wass,
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len
        return loss_dict

    def encode(self, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge,
                num_mol
    ):
        num_graphs = num_mol
        device = node_pos.device

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

       
        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert

            h_node = F.one_hot(node_type, self.num_node_types).float()
            h_halfedge = F.one_hot(halfedge_type, self.num_edge_types).float()
            h_edge = torch.cat([h_halfedge, h_halfedge], dim=0)
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert

            h_node = h_node_0
            h_edge = torch.cat([h_halfedge_0, h_halfedge_0], dim=0)
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # 2.2 embedding
        emb = self.encoder(h_node, node_pos, batch_node, h_edge, edge_index, batch_edge)[0]
        return emb


    def forward(self, h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, t, emb):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        # 1 node and edge embedding + time embedding
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        emb_node = emb.index_select(0, batch_node)
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node, emb_node], dim=-1)
        
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        emb_edge = emb.index_select(0, batch_edge)
        h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge, emb_edge], dim=-1)

        # 2 diffuse to get the updated node embedding and bond embedding
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_pert,
            pos_node=pos_pert, 
            h_edge=h_edge_pert, 
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps,
        )
        
        n_halfedges = h_edge.shape[0] // 2
        pred_node = self.node_decoder(h_node)
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges]+h_edge[n_halfedges:])
        pred_pos = pos_node
        
        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
        }  # at step 0

    @torch.no_grad()
    def sample(self, n_graphs, batch_node, halfedge_index, batch_halfedge, bond_predictor=None, guidance=None, emb=None):
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init
            

        # # 1.5 log init
        node_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps+1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        if emb is None:
            emb = torch.randn(n_graphs, self.emb_dim, device=device)
            
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = self(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, 
                time_step, 
                emb
            )
            pred_node = preds['pred_node'].detach()  # (N, num_node_types)
            pred_pos = preds['pred_pos'].detach()  # (N, 3)
            pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)

            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, batch_node, v0_prob=True)
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, batch_halfedge, v0_prob=True)
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
                
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        h_node_in = h_node_pert.detach()
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        if gui_type == 'entropy':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
                            prob = torch.softmax(pred_bondpredictor, dim=-1)
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log()
                            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'entropy_bond':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log()
                            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'logit_bond':
                            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
                            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'logit':
                            ind_bond_notmask = (halfedge_type_prev <= 4)
                            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent':
                            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
                            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent_bond':
                            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
                            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta

            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev
            halfedge_traj[i+1] = h_halfedge_prev

            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        # # 3. get the final positions
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],
        }
        

    @torch.no_grad()
    def sample_ddim(self, n_graphs, batch_node, halfedge_index, batch_halfedge, batch, inverse=True, stride=1, bond_predictor=None, guidance=None, emb_init=None, emb=None):
        """
        Trajectory storage temporarily disabled
        """
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)

        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        if emb_init is None:
            emb_init = torch.randn(n_graphs, self.emb_dim, device=device)


        time_sequence = list(range(0, self.num_timesteps-stride, stride))
        if inverse:
            h_node_pert = F.one_hot(batch.node_type, self.num_node_types).float()
            pos_pert = batch.node_pos
            h_halfedge_pert = F.one_hot(batch.halfedge_type, self.num_edge_types).float()
            
            for i, step in tqdm(enumerate(time_sequence), total=len(time_sequence)):
                time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
                h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
                
                # # 1 inference
                preds = self(
                    h_node_pert, pos_pert, batch_node,
                    h_edge_pert, edge_index, batch_edge, 
                    time_step, 
                    emb_init
                )
                pred_node = preds['pred_node'].detach()  # (N, num_node_types)
                pred_pos = preds['pred_pos'].detach()  # (N, 3)
                pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)
            
                
                # # 2 get the t - 1 state
                pos_prev = self.pos_transition.reverse_sample_ddim(
                    x_t=pos_pert, x_recon=pred_pos, t=time_step, s=time_step+stride,  batch=batch_node)
                
                h_node_prev = self.node_transition.reverse_sample_ddim(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, s=time_step+stride, batch=batch_node)
                h_halfedge_prev = self.edge_transition.reverse_sample_ddim(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, s=time_step+stride, batch=batch_halfedge)
                
                # # 3 update t-1
                pos_pert = pos_prev
                h_node_pert = h_node_prev
                h_halfedge_pert = h_halfedge_prev

            pos_init = pos_pert
            h_node_init = h_node_pert
            h_halfedge_init = h_halfedge_pert

        else:
            node_init = self.node_transition.sample_init(n_nodes_all)
            pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
            halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
            #if self.categorical_space == 'discrete':
            # DDIM sampling for continuous only
            h_node_init = node_init
            h_halfedge_init = halfedge_init
            

        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init

        if emb is None:
            emb = emb_init.clone()

            
        for i, step in tqdm(enumerate(time_sequence[::-1]), total=len(time_sequence)):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = self(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, 
                time_step, 
                emb
            )
            pred_node = preds['pred_node'].detach()  # (N, num_node_types)
            pred_pos = preds['pred_pos'].detach()  # (N, 3)
            pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)

            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.reverse_sample_ddim(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, s=time_step-stride,  batch=batch_node)
        
            h_node_prev = self.node_transition.reverse_sample_ddim(
                x_t=h_node_pert, x_recon=pred_node, t=time_step, s=time_step-stride, batch=batch_node)
            h_halfedge_prev = self.edge_transition.reverse_sample_ddim(
                x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, s=time_step-stride, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        h_node_in = h_node_pert.detach()
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        if gui_type == 'entropy':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
                            prob = torch.softmax(pred_bondpredictor, dim=-1)
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log()
                            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'entropy_bond':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log()
                            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'logit_bond':
                            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
                            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'logit':
                            ind_bond_notmask = (halfedge_type_prev <= 4)
                            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent':
                            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
                            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent_bond':
                            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
                            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta


            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        # # 3. get the final positions
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
        }


### With properties
class MolEncoderFT(Module):
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,  # explicite bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_timesteps = 0

        # # embedding
        node_dim = config.encoder.node_dim
        edge_dim = config.encoder.edge_dim
        emb_dim = config.encoder.emb_dim
        
        emb_prop_dim = config.encoder.emb_prop_dim
        prop_dim = config.encoder.prop_dim
        self.emb_prop_dim = emb_prop_dim
        
        time_dim = config.diff.time_dim if self.num_timesteps > 0 else 0
        self.node_embedder = nn.Linear(num_node_types, node_dim-time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim-time_dim, bias=False) # the init edge features
        if self.num_timesteps != 0:
            self.time_emb = GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear')
        # # predictor
        self.encoder = NodeEdgeNet(**config.encoder)
        self.edge_decoder = MLP(edge_dim + node_dim, num_edge_types, edge_dim, num_layer=3)
        
        self.edge_weight = torch.tensor([0.1]+[1.]*(self.num_edge_types-1), dtype=torch.float32)
        self.ce_loss = torch.nn.CrossEntropyLoss(self.edge_weight)

        self.final = MLP(node_dim + edge_dim, emb_dim, emb_dim*2, num_layer=2)

        if config.encoder.prop_mlp_layers == 1:
            self.aux_cls = nn.Linear(emb_prop_dim, prop_dim)
        else:
            if config.encoder.prop_residual == 'linear':
                self.aux_cls = MLP(emb_prop_dim, prop_dim, config.encoder.prop_hidden_dim, num_layer=2)
            elif config.encoder.prop_residual == 'res-add':
                print("Using residual connection (add)")
                self.aux_cls = MLPResidualAdd(emb_prop_dim, prop_dim, num_layer=2)
            elif config.encoder.prop_residual == 'res-cat':
                print("Using residual connection (cat)")
                self.aux_cls = MLPResidualCat(emb_prop_dim, prop_dim, config.encoder.prop_hidden_dim, num_layer=2)                

    def forward(self, h_node, pos_node, batch_node, h_edge, edge_index, batch_edge):
        """
        Predict the edge type of edges defined by halfedge_index
        """
        
        # embedding 

        if self.num_timesteps != 0:
            time_embed_node = self.time_emb(t.index_select(0, batch_node))
            h_node = torch.cat([self.node_embedder(h_node), time_embed_node], dim=-1)
            time_embed_node = self.time_emb(t.index_select(0, batch_edge))
            h_edge = torch.cat([self.edge_embedder(h_edge), time_embed_node], dim=-1)
        else:
            h_node = self.node_embedder(h_node)
            h_edge = self.edge_embedder(h_edge)
            t = torch.zeros(batch_node.max()+1, device=pos_node.device, dtype=torch.long)

        h_node, _, h_edge = self.encoder(
            h_node=h_node,
            pos_node=pos_node, 
            h_edge=h_edge, 
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / max(self.num_timesteps, 1),
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / max(self.num_timesteps, 1),
        )

        h_sub = []

        for k in range(batch_node.max()+1):
            h_node_sub = h_node[batch_node==k].mean(0)
            h_edge_sub = h_edge[batch_edge==k].mean(0)
        
            h_sub.append(torch.cat([h_node_sub, h_edge_sub], dim=0))

        h_sub = torch.vstack(h_sub)
        emb = self.final(h_sub)
        pred = self.aux_cls(emb[:, :self.emb_prop_dim])
        return emb, batch_node, pred


class MolDiffAEFT(Module):
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        emb_dim = config.encoder.emb_dim

        self.emb_dim = emb_dim
        self.wass_weight = config.encoder.wass_weight

        node_input_dim = node_dim+emb_dim+time_dim
        edge_input_dim = edge_dim+emb_dim+time_dim
        
        
        self.node_embedder = nn.Linear(num_node_types, node_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )

        # # encoder
        self.encoder = MolEncoderFT(config=config,
                                  num_node_types=num_node_types,
                                  num_edge_types=num_edge_types)
        
        # # denoiser
        if config.denoiser.backbone == 'NodeEdgeNet':
            self.denoiser = NodeEdgeNet(node_input_dim, edge_input_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.node_decoder = MLP(node_input_dim, num_node_types, node_dim)
        self.edge_decoder = MLP(edge_input_dim, num_edge_types, edge_dim)



    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 4., 8.]) # for earlier models: [1,4,8] default; [1,1,1] for noscale
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.num_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        # sample time
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def add_noise(self, node_type, node_pos, batch_node,
                    halfedge_type, halfedge_index, batch_halfedge,
                    num_mol, t, bond_predictor=None, **kwargs):
            num_graphs = num_mol
            device = node_pos.device

            time_step = t * torch.ones(num_graphs, device=device).long()

            # 2.1 perturb pos, node, edge
            pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
            node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
            halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
            # edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
            # batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
            if self.categorical_space == 'discrete':
                h_node_pert, log_node_t, log_node_0 = node_pert
                h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
            else:
                h_node_pert, h_node_0 = node_pert
                h_halfedge_pert, h_halfedge_0 = halfedge_pert
            return [h_node_pert, pos_pert, h_halfedge_pert]

    def get_loss(self, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge,
                num_mol, prop
    ):
        num_graphs = num_mol
        device = node_pos.device

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

       
        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert

            h_node = F.one_hot(node_type, self.num_node_types).float()
            h_halfedge = F.one_hot(halfedge_type, self.num_edge_types).float()
            h_edge = torch.cat([h_halfedge, h_halfedge], dim=0)
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert

            h_node = h_node_0
            h_edge = torch.cat([h_halfedge_0, h_halfedge_0], dim=0)
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # 2.2 embedding
        emb, _, aux_pred = self.encoder(h_node, node_pos, batch_node, h_edge, edge_index, batch_edge)


        # 3. forward to denoise
        preds = self(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, 
            time_step,
            emb
        )
        pred_node = preds['pred_node']
        pred_pos = preds['pred_pos']
        pred_halfedge = preds['pred_halfedge']

        # 4. loss
        # 4.1 pos
        loss_pos = F.mse_loss(pred_pos, node_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(node_pos[bond_index[0]] - node_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_pos[bond_index[0]] - pred_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)
    
        if self.categorical_space == 'discrete':
            # 4.2 node type
            log_node_recon = F.log_softmax(pred_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, batch_node, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, batch_node, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=batch_node)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type
            log_halfedge_recon = F.log_softmax(pred_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, batch_halfedge, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, batch_halfedge, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, 
                            log_edge_post_pred, log_halfedge_0, t=time_step, batch=batch_halfedge)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_node, h_node_0)  * 30
            loss_edge = F.mse_loss(pred_halfedge, h_halfedge_0) * 30

        loss_aux = F.mse_loss(aux_pred, prop)
        loss_wass = kernel_mmd_loss(emb)*self.wass_weight

        # total
        loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0) + (loss_wass if self.wass_weight != 0.0 else 0) + loss_aux
        
        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'loss_wass': loss_wass,
            'loss_aux': loss_aux
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len
        return loss_dict

    def encode(self, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge,
                num_mol
    ):
        num_graphs = num_mol
        device = node_pos.device

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

       
        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert

            h_node = F.one_hot(node_type, self.num_node_types).float()
            h_halfedge = F.one_hot(halfedge_type, self.num_edge_types).float()
            h_edge = torch.cat([h_halfedge, h_halfedge], dim=0)
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert

            h_node = h_node_0
            h_edge = torch.cat([h_halfedge_0, h_halfedge_0], dim=0)
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # 2.2 embedding
        emb, _, pred = self.encoder(h_node, node_pos, batch_node, h_edge, edge_index, batch_edge)
        return emb, pred


    def forward(self, h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, t, emb):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        # 1 node and edge embedding + time embedding
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        emb_node = emb.index_select(0, batch_node)
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node, emb_node], dim=-1)
        
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        emb_edge = emb.index_select(0, batch_edge)
        h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge, emb_edge], dim=-1)

        # 2 diffuse to get the updated node embedding and bond embedding
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_pert,
            pos_node=pos_pert, 
            h_edge=h_edge_pert, 
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps,
        )
        
        n_halfedges = h_edge.shape[0] // 2
        pred_node = self.node_decoder(h_node)
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges]+h_edge[n_halfedges:])
        pred_pos = pos_node
        
        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
        }  # at step 0

    @torch.no_grad()
    def sample(self, n_graphs, batch_node, halfedge_index, batch_halfedge, bond_predictor=None, guidance=None, emb=None):
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init
            

        # # 1.5 log init
        node_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps+1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        if emb is None:
            emb = torch.randn(n_graphs, self.emb_dim, device=device)
            
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = self(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, 
                time_step, 
                emb
            )
            pred_node = preds['pred_node'].detach()  # (N, num_node_types)
            pred_pos = preds['pred_pos'].detach()  # (N, 3)
            pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)

            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, batch_node, v0_prob=True)
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, batch_halfedge, v0_prob=True)
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
                
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        h_node_in = h_node_pert.detach()
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        if gui_type == 'entropy':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
                            prob = torch.softmax(pred_bondpredictor, dim=-1)
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log()
                            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'entropy_bond':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log()
                            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'logit_bond':
                            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
                            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'logit':
                            ind_bond_notmask = (halfedge_type_prev <= 4)
                            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent':
                            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
                            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent_bond':
                            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
                            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta

            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev
            halfedge_traj[i+1] = h_halfedge_prev

            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        # # 3. get the final positions
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],
        }
        

    @torch.no_grad()
    def sample_ddim(self, n_graphs, batch_node, halfedge_index, batch_halfedge, batch, inverse=True, stride=1, bond_predictor=None, guidance=None, emb_init=None, emb=None):
        """
        Trajectory storage temporarily disabled
        """
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)

        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        if emb_init is None:
            emb_init = torch.randn(n_graphs, self.emb_dim, device=device)


        time_sequence = list(range(0, self.num_timesteps-stride, stride))
        if inverse:
            h_node_pert = F.one_hot(batch.node_type, self.num_node_types).float()
            pos_pert = batch.node_pos
            h_halfedge_pert = F.one_hot(batch.halfedge_type, self.num_edge_types).float()
            
            for i, step in tqdm(enumerate(time_sequence), total=len(time_sequence)):
                time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
                h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
                
                # # 1 inference
                preds = self(
                    h_node_pert, pos_pert, batch_node,
                    h_edge_pert, edge_index, batch_edge, 
                    time_step, 
                    emb_init
                )
                pred_node = preds['pred_node'].detach()  # (N, num_node_types)
                pred_pos = preds['pred_pos'].detach()  # (N, 3)
                pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)
            
                
                # # 2 get the t - 1 state
                pos_prev = self.pos_transition.reverse_sample_ddim(
                    x_t=pos_pert, x_recon=pred_pos, t=time_step, s=time_step+stride,  batch=batch_node)
                
                h_node_prev = self.node_transition.reverse_sample_ddim(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, s=time_step+stride, batch=batch_node)
                h_halfedge_prev = self.edge_transition.reverse_sample_ddim(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, s=time_step+stride, batch=batch_halfedge)
                
                # # 3 update t-1
                pos_pert = pos_prev
                h_node_pert = h_node_prev
                h_halfedge_pert = h_halfedge_prev

            pos_init = pos_pert
            h_node_init = h_node_pert
            h_halfedge_init = h_halfedge_pert

        else:
            node_init = self.node_transition.sample_init(n_nodes_all)
            pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
            halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
            #if self.categorical_space == 'discrete':
            # DDIM sampling for continuous only
            h_node_init = node_init
            h_halfedge_init = halfedge_init
            

        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init

        if emb is None:
            emb = emb_init.clone()

            
        for i, step in tqdm(enumerate(time_sequence[::-1]), total=len(time_sequence)):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = self(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, 
                time_step, 
                emb
            )
            pred_node = preds['pred_node'].detach()  # (N, num_node_types)
            pred_pos = preds['pred_pos'].detach()  # (N, 3)
            pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)

            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.reverse_sample_ddim(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, s=time_step-stride,  batch=batch_node)
        
            h_node_prev = self.node_transition.reverse_sample_ddim(
                x_t=h_node_pert, x_recon=pred_node, t=time_step, s=time_step-stride, batch=batch_node)
            h_halfedge_prev = self.edge_transition.reverse_sample_ddim(
                x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, s=time_step-stride, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        h_node_in = h_node_pert.detach()
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        if gui_type == 'entropy':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
                            prob = torch.softmax(pred_bondpredictor, dim=-1)
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log()
                            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                        elif gui_type == 'entropy_bond':
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
                            entropy = entropy.log()
                            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'logit_bond':
                            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
                            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'logit':
                            ind_bond_notmask = (halfedge_type_prev <= 4)
                            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
                            pred = pred_real_bond.sum()
                            delta = + torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent':
                            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
                            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == 'crossent_bond':
                            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
                            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
                            entropy = entropy.log().sum()
                            delta = - torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta


            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        # # 3. get the final positions
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
        }
        
def kernel_mmd_loss(X, Y = None, sigma = 1.0):
    if Y is None:
        Y = torch.randn(X.shape[0], X.shape[1], device=X.device)
    n = (X.shape[0] // 2) * 2
    gamma = 1 / (2 * X.shape[-1] * sigma**2)
    rbf = lambda A, B: torch.exp(-gamma * ((A - B) ** 2).sum(-1))
    mmd2 = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
      - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2])).mean()

    return mmd2