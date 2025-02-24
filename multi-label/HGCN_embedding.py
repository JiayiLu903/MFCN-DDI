import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pandas as pd



device = torch.device("cpu")
print(f"Using device: {device}")


print("Reading input data files...")
node_features_df = pd.read_csv('KG_entity.csv', header=None)
relation_features_df = pd.read_csv('KG_relation.csv', header=None)
triplets_df = pd.read_csv('train.tsv', sep='\t', header=None)
print("Data files read successfully.")




def build_graph(triplets_df):
    print("Building graph...")
    g = dgl.DGLGraph()

    heads = triplets_df.iloc[:, 0].values
    tails = triplets_df.iloc[:, 2].values
    relations = triplets_df.iloc[:, 1].values

    all_nodes = pd.concat([triplets_df.iloc[:, 0], triplets_df.iloc[:, 2]]).unique()
    g.add_nodes(len(all_nodes), data={'node_id': torch.tensor(all_nodes, dtype=torch.long).to(device)})
    print(f"Total nodes added: {len(all_nodes)}")

    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}


    # for idx, node_id in enumerate(all_nodes):
    #     print(f"Node Index: {idx}, Node ID: {node_id}")

    g.add_edges(heads, tails)
    print(f"Total edges added: {len(heads)}")

    g.edata['relation'] = torch.tensor(relations, dtype=torch.long).to(device)
    print("Edge relations set.")
    print("Graph building completed.")


    if 'relation' in g.edata:
        print("Graph contains 'relation' edge feature.")
    else:
        print("Graph does not contain 'relation' edge feature.")




    return g, node_id_to_idx


g, node_id_to_idx = build_graph(triplets_df)



print("Initializing node features...")
node_ids = node_features_df.iloc[:, 0].values
num_nodes = len(node_ids)
num_features = node_features_df.shape[1] - 1
node_features = torch.zeros((num_nodes, num_features), dtype=torch.float32).to(device)

for node_id in node_ids:
    if node_id in node_id_to_idx:
        graph_idx = node_id_to_idx[node_id]
        node_features[graph_idx] = torch.tensor(
            node_features_df[node_features_df.iloc[:, 0] == node_id].iloc[:, 1:].values[0], dtype=torch.float32).to(device)

print(f"Node features shape: {node_features.shape}")


print("Initializing relation features...")
relation_ids = relation_features_df.iloc[:, 0].values
num_relations = len(relation_ids)
num_relation_features = relation_features_df.shape[1] - 1
relation_features = torch.zeros((num_relations, num_relation_features), dtype=torch.float32).to(device)

for i, relation_id in enumerate(relation_ids):
    relation_features[i] = torch.tensor(
        relation_features_df[relation_features_df.iloc[:, 0] == relation_id].iloc[:, 1:].values[0], dtype=torch.float32).to(device)

print(f"Relation features shape: {relation_features.shape}")




class HGCNLayer(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(HGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feat_dim, out_feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, g, node_features, batch_nodes):
        with g.local_scope():
            g.ndata['h'] = node_features.to(device)


            new_node_features = torch.zeros(len(batch_nodes), node_features.size(1)).to(device)

            for i, node in enumerate(batch_nodes):

                neighbors = g.successors(node)
                neighbors = torch.cat((torch.tensor([node], device=device), neighbors))


                subgraph = g.subgraph(neighbors.tolist())


                print(f"Node: {node}")
                print("Subgraph nodes:", subgraph.nodes())
                print("Subgraph edges:", subgraph.edges())


                subgraph.ndata['h'] = g.ndata['h'][neighbors].to(device)


                subgraph.ndata['degree'] = subgraph.in_degrees().float().clamp(min=1).to(device)
                subgraph.ndata['c_i'] = 1.0 / torch.sqrt(subgraph.ndata['degree'])



                def message_func(edges):
                    src_h = edges.src['h']
                    degree_src = edges.src['degree']
                    degree_dst = edges.dst['degree']


                    c_ij = torch.where((degree_src == 0) | (degree_dst == 0),
                                       torch.tensor(0.0, device=degree_src.device),
                                       1.0 / torch.sqrt(degree_src * degree_dst))


                    msg = c_ij.view(-1, 1) * torch.matmul(src_h, self.weight)
                    print("msg shape:", msg.shape)
                    print("msg :", msg)
                    return {'msg': msg}

                def reduce_func(nodes):
                    if len(nodes.mailbox['msg']) == 0:
                        return {'h_neigh': torch.zeros_like(nodes.data['h'])}
                    h_neigh = torch.sum(nodes.mailbox['msg'], dim=1)
                    return {'h_neigh': h_neigh}

                subgraph.update_all(message_func, reduce_func)

                print("After update_all, subgraph.ndata keys:", subgraph.ndata.keys())

                if 'h_neigh' in subgraph.ndata:
                    print("'h_neigh' shape:", subgraph.ndata['h_neigh'].shape)
                    print("'h_neigh':", subgraph.ndata['h_neigh'])
                else:
                    print("Error: 'h_neigh' was not found in subgraph.ndata")


                def apply_node_func(nodes):
                    if 'h_neigh' in nodes.data and len(nodes.data['h_neigh']) > 0:
                        h_neigh_sum = torch.sum(nodes.data['h_neigh'], dim=0, keepdim=True)
                    else:
                        h_neigh_sum = torch.zeros_like(nodes.data['h'])

                    h_self = nodes.data['h']
                    c_i = nodes.data['c_i'].unsqueeze(-1)


                    h_new = F.relu(h_neigh_sum + c_i * h_self)


                    print(f"Current node: {nodes}")
                    print(f"Current node self feature (h_self): {h_self}")
                    print(f"Sum of neighbor features (h_neigh_sum): {h_neigh_sum}")
                    print(f"Updated node feature (h_new): {h_new}")

                    return {'h_new': h_new}


                subgraph.apply_nodes(apply_node_func)


                matching_indices = (subgraph.ndata[dgl.NID] == node).nonzero(as_tuple=True)[0]

                if matching_indices.numel() > 0:
                    node_idx_in_subgraph = matching_indices[0].item()
                else:
                    raise ValueError(f"No matching index found for node: {node} in the subgraph.")

                new_node_features[i] = subgraph.ndata['h_new'][node_idx_in_subgraph]


                print(f"New node features for node {i}: {new_node_features[i]}")


            return new_node_features




target_node_ids = torch.arange(0, 1911, dtype=torch.long).to(device)
target_node_indices = torch.tensor([node_id_to_idx[node_id.item()] for node_id in target_node_ids if node_id.item() in node_id_to_idx], dtype=torch.long).to(device)


class HGCN(nn.Module):
    def __init__(self, in_feat_dim):
        super(HGCN, self).__init__()

        self.layer = HGCNLayer(in_feat_dim, in_feat_dim)

    def forward(self, g, node_features, target_nodes, batch_size=256):
        print("Starting forward pass in HGCN model...")
        num_nodes = target_nodes.size(0)
        final_node_features = torch.zeros((num_nodes, node_features.size(1)), dtype=torch.float32).to(device)


        for batch_start in range(0, num_nodes, batch_size):
            batch_end = min(batch_start + batch_size, num_nodes)
            batch_nodes = target_nodes[batch_start:batch_end]


            batch_node_features = self.layer(g, node_features, batch_nodes)


            final_node_features[batch_start:batch_end] = batch_node_features

        print("Forward pass in HGCN model completed.")
        return final_node_features


print("Initializing model...")
in_feat_dim = node_features.size(1)
model = HGCN(in_feat_dim).to(device)
batch_size=256


print("Starting forward propagation for target nodes...")
final_node_features = model(g, node_features, target_node_indices, batch_size=batch_size)


del node_features, g, model
torch.cuda.empty_cache()


print("Final node features calculated.")
print(final_node_features)
print("Final node features shape:", final_node_features.shape)


print("Saving final node features to CSV file...")

final_node_features_np = final_node_features.detach().cpu().numpy()

output_filename = "final_node_features.csv"
pd.DataFrame(final_node_features_np).to_csv(output_filename, header=False, index=False)
print(f"Final node features saved to {output_filename}")


