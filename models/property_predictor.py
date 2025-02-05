import torch
from torch.nn import Module
from torch.nn import functional as F

from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet
from .common import *
from .diffusion import *


class PropertyPredictor(Module):
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
        self.define_betas_alphas(config.diff)
        if hasattr(config,'output_dim'):
            self.output_dim = config.output_dim
        else:
            self.output_dim = 1

        # # embedding
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim if self.num_timesteps > 0 else 0
        self.node_embedder = nn.Linear(num_node_types, node_dim-time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_node_types * 2, edge_dim-time_dim, bias=False) # the init edge features
        if self.num_timesteps != 0:
            self.time_emb = GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear')
        # # predictor
        self.encoder = NodeEdgeNet(node_dim, edge_dim, **config.encoder)
        self.edge_decoder = MLP(edge_dim + node_dim, num_edge_types, edge_dim, num_layer=3)
        
        self.mse_loss = torch.nn.MSELoss()

        self.final = MLP(node_dim + edge_dim, self.output_dim, node_dim + edge_dim, num_layer=2)
        
        self.predict_mode = getattr(config, 'predict_mode', 'regression')
        self.label_weight = getattr(config, 'label_weight', None)
        if self.predict_mode == 'regression':
            self.pred_loss = nn.MSELoss()

        elif self.predict_mode == 'binary':
            self.pred_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(self.label_weight))
    
    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        if self.num_timesteps == 0:
            return
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 4., 8.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (define for compatibility)

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
            # init_prob = getattr(config.diff_atom, 'init_prob', None)
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.num_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
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

    def get_loss(self, node_type, node_pos, batch_node,
                 halfedge_type, halfedge_index, batch_halfedge, prop,
                num_mol,
    ):
        num_graphs = num_mol
        device = node_pos.device
        if self.num_timesteps != 0:
            time_step, _ = self.sample_time(num_graphs, device)
        else:
            time_step = None # torch.zeros(num_graphs, device=device, dtype=torch.long)

        # 2.1 prepare node hidden  (can be compatible for discrete and continuous)
        if self.num_timesteps != 0:
            pos_node = self.pos_transition.add_noise(node_pos, time_step, batch_node)
            node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
            h_node = node_pert[0]  # compatible for both discrete and continuous catergorical_space
        else:
            h_node = F.one_hot(node_type, self.num_node_types).float()
            pos_node = node_pos
        
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        # 3. forward to denoise
        pred_prop = self(h_node, pos_node, batch_node,
                             edge_index, batch_edge,
                             time_step)
        # pred_halfedge = pred_edge[:halfedge_index.shape[1]]

        # 4. loss
        # 4.3 edge type
        loss_prop = self.pred_loss(pred_prop, prop)

        # total
        loss_total = loss_prop
        
        loss_dict = {
            'loss': loss_total,
            'loss_prop': loss_prop,
        }
        return loss_dict
    


    def forward(self, h_node, pos_node, batch_node,
                edge_index, batch_edge, t):
        """
        Predict the edge type of edges defined by halfedge_index
        """
        
        # embedding 
        h_edge = torch.cat([h_node[edge_index[0]], h_node[edge_index[1]]], dim=-1)
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
        pred = self.final(h_sub)
        
        return pred


class PropertyPredictorWithEdge(Module):
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
        self.define_betas_alphas(config.diff)
        if hasattr(config,'output_dim'):
            self.output_dim = config.output_dim
        else:
            self.output_dim = 1
            
        # # embedding
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim if self.num_timesteps > 0 else 0
        self.node_embedder = nn.Linear(num_node_types, node_dim-time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # bond type
        if self.num_timesteps != 0:
            self.time_emb = GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear')
        # # predictor
        self.encoder = NodeEdgeNet(node_dim, edge_dim, **config.encoder)
        self.edge_decoder = MLP(edge_dim + node_dim, num_edge_types, edge_dim, num_layer=3)
        
        self.mse_loss = torch.nn.MSELoss()
        self.predict_mode = getattr(config, 'predict_mode', 'regression')
        self.label_weight = getattr(config, 'label_weight', None)
        if self.predict_mode == 'regression':
            self.pred_loss = nn.MSELoss()

        elif self.predict_mode == 'binary':
            self.pred_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(self.label_weight))

        self.final = MLP(node_dim + edge_dim, self.output_dim, node_dim + edge_dim, num_layer=2)

    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        if self.num_timesteps == 0:
            return
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 4., 8.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (define for compatibility)

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
            # init_prob = getattr(config.diff_atom, 'init_prob', None)
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

    def get_loss(self, node_type, node_pos, batch_node,
                 halfedge_type, halfedge_index, batch_halfedge, prop,
                num_mol,
    ):
        num_graphs = num_mol
        device = node_pos.device
        if self.num_timesteps != 0:
            time_step, _ = self.sample_time(num_graphs, device)
        else:
            time_step = None # torch.zeros(num_graphs, device=device, dtype=torch.long)

        # 2.1 prepare node hidden  (can be compatible for discrete and continuous)
        if self.num_timesteps != 0:
            pos_node = self.pos_transition.add_noise(node_pos, time_step, batch_node)
            node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
            h_node = node_pert[0]  # compatible for both discrete and continuous catergorical_space
            halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
            h_halfedge = halfedge_pert[0]
        else:
            h_node = F.one_hot(node_type, self.num_node_types).float()
            h_halfedge = F.one_hot(halfedge_type, self.num_edge_types).float()
            pos_node = node_pos
            
        #h_halfedge = F.one_hot(halfedge_type, self.num_edge_types).float()
        
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        
        h_edge = torch.cat([h_halfedge, h_halfedge], dim=0)

        # 3. forward to denoise
        pred_prop = self(h_node, pos_node, batch_node, h_edge,
                             edge_index, batch_edge,
                             time_step)
        # pred_halfedge = pred_edge[:halfedge_index.shape[1]]

        # 4. loss
        # 4.3 edge type
        loss_prop = self.pred_loss(pred_prop, prop)

        # total
        loss_total = loss_prop
        
        loss_dict = {
            'loss': loss_total,
            'loss_prop': loss_prop,
        }
        return loss_dict


    def forward(self, h_node, pos_node, batch_node, h_edge,
                edge_index, batch_edge, t):
        """
        Predict the edge type of edges defined by halfedge_index
        """
        
        # embedding 
        #h_edge = torch.cat([h_node[edge_index[0]], h_node[edge_index[1]]], dim=-1)
        if self.num_timesteps != 0:
            time_embed_node = self.time_emb(t.index_select(0, batch_node))
            h_node = torch.cat([self.node_embedder(h_node), time_embed_node], dim=-1)
            time_embed_node = self.time_emb(t.index_select(0, batch_edge))
            #h_edge = torch.cat([self.edge_embedder(h_edge), time_embed_node], dim=-1)
        else:
            h_node = self.node_embedder(h_node)
            #h_edge = self.edge_embedder(h_edge)
            t = torch.zeros(batch_node.max()+1, device=pos_node.device, dtype=torch.long)

        h_edge = self.edge_embedder(h_edge)
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
        pred = self.final(h_sub)
        
        return pred