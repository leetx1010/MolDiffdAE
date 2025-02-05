
DEFAULT_FOLLOW_BATCH = ['protein_element', 'ligand_context_element',]

def seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge):
    outputs_pred = outputs['pred']
    outputs_traj = outputs['traj']

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_halfedge = (batch_halfedge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
        new_pred_this = [outputs_pred[0][ind_node],  # node type
                         outputs_pred[1][ind_node],  # node pos
                         outputs_pred[2][ind_halfedge]]  # halfedge type
                        
        new_traj_this = [outputs_traj[0][:, ind_node],  # node type. The first dim is time
                         outputs_traj[1][:, ind_node],  # node pos
                         outputs_traj[2][:, ind_halfedge]]  # halfedge type
        
        halfedge_index_this = halfedge_index[:, ind_halfedge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        new_outputs.append({
            'pred': new_pred_this,
            'traj': new_traj_this,
            'halfedge_index': halfedge_index_this,
        })
    return new_outputs


def seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge):
    outputs_pred = outputs['pred']

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_halfedge = (batch_halfedge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
        new_pred_this = [outputs_pred[0][ind_node],  # node type
                         outputs_pred[1][ind_node],  # node pos
                         outputs_pred[2][ind_halfedge]]  # halfedge type
                        
        halfedge_index_this = halfedge_index[:, ind_halfedge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        ### Standardized output format (TXL)
        new_outputs.append({
            'pred': new_pred_this,
            'halfedge_index': halfedge_index_this,
        })
        """
        new_outputs.append({
            'node': new_pred_this[0],
            'pos': new_pred_this[1],
            'halfedge': new_pred_this[2],
            'halfedge_index': halfedge_index_this,
        })
        """
    return new_outputs

def process_outputs(outputs, batch_node_raw, halfedge_index_raw, batch_halfedge_raw):
    batch_node, halfedge_index, batch_halfedge = batch_node_raw.cpu().numpy(), halfedge_index_raw.cpu().numpy(), batch_halfedge_raw.cpu().numpy()
    output_list = seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)

    gen_list = []
    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    for i_mol, output_mol in enumerate(output_list):
        mol_info = featurizer.decode_output(
            pred_node=output_mol['pred'][0],
            pred_pos=output_mol['pred'][1],
            pred_halfedge=output_mol['pred'][2],
            halfedge_index=output_mol['halfedge_index'],
        )  # note: traj is not used
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
        except MolReconsError:
            pool.failed.append(mol_info)
            continue
        mol_info['rdmol'] = rdmol
        smiles = Chem.MolToSmiles(rdmol)
        mol_info['smiles'] = smiles
        if '.' in smiles:
            pool.failed.append(mol_info)
        else:   # Pass checks!
            gen_list.append(mol_info) 

    return gen_list