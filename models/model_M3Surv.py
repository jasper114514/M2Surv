import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.cross_attention import FeedForward, MMAttentionLayer
from models.layers.fusion import AlignFusion, ShareGuide

from sklearn.cluster import KMeans
from .util import initialize_weights, NystromAttention, SNN_Block
from torch_geometric.nn import GCNConv, GATConv
import dhg
from dhg.nn import HGNNPConv

from collections import defaultdict

def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class Prototype(nn.Module):
    def __init__(self,
                 x_dim = 256,
                 z_dim = 256,
                 beta = 1e-2,
                 sample_num = 50,
                 topk = 256,
                 num_classes = 4,
                 seed = 1):
        super(Prototype, self).__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim
        
        self.encoder = nn.Linear(z_dim, z_dim)
        # decoder a simple logistic regression as in the paper
        self.decoder_logits = nn.Linear(z_dim, num_classes)

        # design proxies for histology images
        self.proxies = nn.Parameter(torch.empty([num_classes*2, z_dim*2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)

    def gaussian_noise(self, samples, K, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K),
                                generator=torch.manual_seed(seed)).cuda() # must be the same seed as the training seed


    def encoder_proxies(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:]) # Make sigma always positive

        return mu_proxy, sigma_proxy

    def forward(self,x, y=None, c=None):
        mu_proxy, sigma_proxy = self.encoder_proxies()

        # calculate z
        z = self.encoder(x)

        # print(z.shape)
        z_norm = F.normalize(z, dim=2)
        mu_proxy_norm = F.normalize(mu_proxy, dim=1)
    
        att_scores = torch.matmul(z_norm, mu_proxy_norm.T)  # z,mu_proxy: (batch_size, feature_dim)

        att_weights = F.softmax(att_scores, dim=-1)
        dynamic_mu = torch.matmul(att_weights, mu_proxy)
    
        return dynamic_mu

class M3Surv(nn.Module):
    def __init__(self, genomic_sizes=[], transomic_sizes=[], n_classes=4, fusion="concat", model_size="small",graph_type="HGNN"):
        super(M3Surv, self).__init__()
        self.graph_type = graph_type
        self.genomic_sizes = genomic_sizes
        self.transomic_sizes = transomic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_pathways = len(transomic_sizes)
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024,256,  256], "large": [1024, 1024, 1024, 256]},
        }
        
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        
        fc_ff = []
        for idx in range(len(hidden) - 1):
            fc_ff.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            
            fc_ff.append(nn.ReLU6())
            fc_ff.append(nn.Dropout(0.25))
        self.ff_fc = nn.Sequential(*fc_ff)

        fc_ffpe = []
        for idx in range(len(hidden) - 1):
            fc_ffpe.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            
            fc_ffpe.append(nn.ReLU6())
            fc_ffpe.append(nn.Dropout(0.25))
        self.ffpe_fc = nn.Sequential(*fc_ffpe)
        
        # Genomic Embedding Network
        # self.init_per_gene_model(genomic_sizes)
        self.init_per_trans_model(transomic_sizes)
        
        self.graph_ff = [HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, use_bn = True, drop_rate=0.25).to("cuda")]
        self.graph_ffpe = [HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, use_bn = True, drop_rate=0.25).to("cuda")]
        self.graph_share = [HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, use_bn = True, drop_rate=0.25).to("cuda")]
        # self.graph_gene = [HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
        #                                 HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
        #                                 HGNNPConv(256, 256, use_bn = True, drop_rate=0.25).to("cuda")]
        self.graph = [GCNConv(256, 256, 1).to("cuda"),
                        GCNConv(256, 256, 1).to("cuda"),
                        GCNConv(256, 256, 1).to("cuda")]
        # multi-modal fusion
        self.attention_fusion = AlignFusion(
            embedding_dim=256,
            num_heads=8
        )
        
        self.ff_weight = nn.Linear(hidden[idx], 1)
        self.ffpe_weight = nn.Linear(hidden[idx], 1)
        
        
        self.share_guide = ShareGuide(
            embedding_dim=256,
            num_heads=8
        )
        # Classification Layer
        self.feed_forward = FeedForward(256, dropout=0.25)
        self.layer_norm = nn.LayerNorm(256)
        self.p_norm = nn.LayerNorm(256)
        
        self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1]*2,hidden[-1]//2), nn.ReLU6()]# , nn.Linear(hidden[-1],hidden[-1]//2), nn.ReLU6()]
            )
        
        self.classifier = nn.Linear(hidden[-1]//2, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_ff = kwargs["ff_path"]
        x_ffpe = kwargs["ffpe_path"]
        graph = kwargs["graph"][0]
        # adj = kwargs["adj"]
        missing = True
        if x_ff is not None and x_ffpe is not None and kwargs.get("x_genomic1") is not None:
            missing = False
            # x_genomic = [kwargs["x_genomic%d" % i].to("cuda") for i in range(1, 7)]
            x_transomic = [kwargs["x_transomic%d" % i].to("cuda") for i in range(1, self.num_pathways+1)]
        
            ffpe_features = self.ffpe_fc(x_ffpe)[0]
            ff_features = self.ff_fc(x_ff)[0]
            
            ff_weight = torch.mean(self.ff_weight(ff_features))
            ffpe_weight = torch.mean(self.ffpe_weight(ffpe_features))
            
            ff_weight_sigmoid = torch.sigmoid(ff_weight)
            ffpe_weight_sigmoid = torch.sigmoid(ffpe_weight)

            ff_weight_sigmoid = ff_weight_sigmoid + torch.log(ff_weight_sigmoid)/(torch.log(ff_weight_sigmoid*ffpe_weight_sigmoid)+1e-8)
            ffpe_weight_sigmoid = ffpe_weight_sigmoid + torch.log(ffpe_weight_sigmoid)/(torch.log(ff_weight_sigmoid*ffpe_weight_sigmoid)+1e-8)

            weights_concat = torch.stack([ff_weight_sigmoid, ffpe_weight_sigmoid])
            weights = F.softmax(weights_concat, dim=0)
            p_features = torch.cat((ff_features, ffpe_features), dim=-2)
            
            # print(weights)
            if self.graph_type == "HGNN":
                ff_hyper_index = self.get_hyperedge(graph.ff_edge_index)
                ff_hyper_latent = self.get_hyperedge(graph.ff_edge_latent)
                ff_hg = dhg.Hypergraph(num_v=ff_features.shape[0], e_list=ff_hyper_index+ff_hyper_latent) # ff_hyper_index+ff_hyper_latent

                ffpe_hyper_index = self.get_hyperedge(graph.ffpe_edge_index)
                ffpe_hyper_latent = self.get_hyperedge(graph.ffpe_edge_latent)
                ffpe_hg = dhg.Hypergraph(num_v=ffpe_features.shape[0], e_list=ffpe_hyper_index+ffpe_hyper_latent)

                share_hyper_index = self.get_hyperedge(graph.share_edge)
                share_hg = dhg.Hypergraph(num_v=p_features.shape[0], e_list=share_hyper_index)
                
                
            
            for l in range(0,3):
                ff_features = self.graph_ff[l](ff_features, ff_hg)
                ffpe_features = self.graph_ffpe[l](ffpe_features, ffpe_hg)
                p_features = self.graph_share[l](p_features, share_hg)
            
                    
            ff_features = ff_features.unsqueeze(0)
            ffpe_features = ffpe_features.unsqueeze(0)
            p_features = p_features.unsqueeze(0)
            ff, ffpe = self.share_guide(ff_features, ffpe_features, p_features)
                
            pathology_features = torch.cat((ff, ffpe), dim=-2)
            
            # gene 
            transomic = [self.trans_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_transomic)] 
            genomics_features = torch.stack(transomic).unsqueeze(0)
            proteomic = kwargs["proteomic"]
            if proteomic is not None:
                proteomic = self.protein_SNN(proteomic.to('cuda'))
                genomics_features = torch.cat((torch.stack(genomic), torch.stack(transomic), proteomic), dim=0).unsqueeze(0) # torch.stack(transomic).unsqueeze(0) [6+331,256] # 
            else: 
                genomics_features = torch.cat((torch.stack(genomic), torch.stack(transomic)), dim=0).unsqueeze(0)
            p_pro = self.build_prototypes(pathology_features, n_prototypes=32)
            g_pro = self.build_prototypes(genomics_features, n_prototypes=16)
            
            pathology_memory = pathology_features.clone()
            genomics_memory = genomics_features.clone()
            
        elif x_ff is not None and x_ffpe is not None:
            # gene missing
            bank = kwargs["memory"]
            ff_features = self.ff_fc(x_ff)[0]
            ffpe_features = self.ffpe_fc(x_ffpe)[0]
            p_features = torch.cat((ff_features, ffpe_features), dim=-2)
            
            if self.graph_type == "HGNN":
                ff_hyper_index = self.get_hyperedge(graph.ff_edge_index)
                ff_hyper_latent = self.get_hyperedge(graph.ff_edge_latent)
                ff_hg = dhg.Hypergraph(num_v=ff_features.shape[0], e_list=ff_hyper_index+ff_hyper_latent)

                ffpe_hyper_index = self.get_hyperedge(graph.ffpe_edge_index)
                ffpe_hyper_latent = self.get_hyperedge(graph.ffpe_edge_latent)
                ffpe_hg = dhg.Hypergraph(num_v=ffpe_features.shape[0], e_list=ffpe_hyper_index+ffpe_hyper_latent)

                share_hyper_index = self.get_hyperedge(graph.share_edge)
                share_hg = dhg.Hypergraph(num_v=p_features.shape[0], e_list=share_hyper_index)
            
            for l in range(0,3):
                ff_features = self.graph_ff[l](ff_features, ff_hg)
                ffpe_features = self.graph_ffpe[l](ffpe_features, ffpe_hg)
                p_features = self.graph_share[l](p_features, share_hg)
            
            ff_features = ff_features.unsqueeze(0)
            ffpe_features = ffpe_features.unsqueeze(0)
            p_features = p_features.unsqueeze(0)
            
            ff, ffpe = self.share_guide(ff_features, ffpe_features, p_features)
            pathology_features = torch.cat((ff, ffpe), dim=-2)
            genomics_features = bank.retrieveGene(self.build_prototypes(pathology_features, 32))
            
        else:
            bank = kwargs["memory"]
            # pathology missing
            x_genomic = [kwargs["x_genomic%d" % i].to("cuda") for i in range(1, 7)]
            x_transomic = [kwargs["x_transomic%d" % i].to("cuda") for i in range(1, self.num_pathways+1)]
            proteomic = kwargs["proteomic"]
            genomic = [self.gene_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_genomic)]
            transomic = [self.trans_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_transomic)] 
            if proteomic is not None:
                proteomic = self.protein_SNN(proteomic.to('cuda'))
                genomics_features = torch.cat((torch.stack(genomic), torch.stack(transomic), proteomic), dim=0).unsqueeze(0) # torch.stack(transomic).unsqueeze(0) [6+331,256] # 
            else: 
                genomics_features = torch.cat((torch.stack(genomic), torch.stack(transomic)), dim=0).unsqueeze(0)
            pathology_features = bank.retrievePathology(self.build_prototypes(genomics_features, 16))
            
           
        g_total = genomics_features.shape[1]
        
        token = torch.cat((genomics_features, pathology_features), dim=1)
        
        token_cross = self.attention_fusion(token, g_num=g_total)
        
        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)
       
        gene_embed = token_cross[:, :g_total, :]
        gene_embed = torch.mean(gene_embed, dim=1)
        
        path_embed = token_cross[:, g_total:, :]
        path_embed = torch.mean(path_embed, dim=1)
        
        fusion = self.mm(
                torch.cat([gene_embed, path_embed], dim=1)
            )  
        
        logits = self.classifier(fusion)  # [1, n_classes]
        
        if missing:
            return logits, logits
        else:
            return logits, {'pathology_feature':pathology_memory, 'pathology_prototype':p_pro, 'genomics_feature':genomics_memory, 'genomics_prototype':g_pro}

    def init_per_gene_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.gene_sig_networks = nn.ModuleList(sig_networks)
        
    def init_per_trans_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.trans_sig_networks = nn.ModuleList(sig_networks)
    
    def get_hyperedge(self, edge):
        
        adj_matrix = edge.cpu().numpy()

        hyperedges = defaultdict(set)
        
        for start, end in adj_matrix.T:
            hyperedges[start].add(end)

        hypergraph_edges = [list({start}.union(ends)) for start, ends in hyperedges.items()]

        return hypergraph_edges
    
    def build_prototypes(self, features, n_prototypes=10):
        features = features.squeeze(0)
        kmeans = KMeans(n_clusters=n_prototypes)
        kmeans.fit(features.detach().cpu().numpy())
        return torch.tensor(kmeans.cluster_centers_, device='cuda')  # [n_prototypes, feat_dim]
    
    def binary_edge(self, edge):
    
        edge = edge.to('cuda')

        edges = edge.t()

        unique_edges_tensor = torch.unique(edges, dim=0)
        return unique_edges_tensor.t()