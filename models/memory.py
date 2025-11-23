import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, path='memory', theta = 0.0):
        self.path = path
        self.theta = theta
    
    def clear(self):    
        file_path = self.path
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    def save(self, memory_dict_list):
        with h5py.File(self.path, 'w') as hf:
            for i, patient_data in enumerate(memory_dict_list):
                group = hf.create_group(f"patient_{i}")
            
                for key in ['pathology_feature', 'genomics_feature']:
                    data = patient_data[key]

                    if key == 'pathology_feature' and isinstance(data, torch.Tensor):
                        target_len = 4096
                        N, C = data.shape
                    
                        if N >= target_len:
                            data = data[-target_len:, :]
                        elif N > 0:
                            indices = torch.arange(target_len, device=data.device) % N
                            data = data[indices]
                        else:
                            data = torch.zeros(target_len, C, device=data.device, dtype=data.dtype)

                    if isinstance(data, torch.Tensor):
                        data = data.detach().cpu().numpy()
                    group.create_dataset(key, data=data)
    
    def retrievePathology(self, g_feat):
        best_similarity = -1
        best_pathology = None
        g_feat = g_feat.detach().cpu()

        if not isinstance(g_feat, torch.Tensor):
            g_feat = torch.tensor(g_feat)
        
        with h5py.File(self.path, 'r') as hf:
            for patient in hf:
                patient_data = hf[patient]
                stored_g_feat = torch.tensor(patient_data['genomics_feature'][:])
                        
                similarity = F.cosine_similarity(
                    g_feat.flatten().unsqueeze(0),
                    stored_g_feat.flatten().unsqueeze(0)
                ).item()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pathology = torch.tensor(patient_data['pathology_feature'][:])
        
        return best_pathology.to('cuda')
    
    def retrieveGene(self, p_feat):
        best_similarity = -1
        best_gene = None
        p_feat = p_feat.detach().cpu()
        target_len = 4096
        N, C = p_feat.shape
                    
        if N >= 4096:
            p_feat = p_feat[-target_len:, :]
        elif N > 0:
            indices = torch.arange(target_len, device=p_feat.device) % N
            p_feat = p_feat[indices]
        
        with h5py.File(self.path, 'r') as hf:
            for patient in hf:
                patient_data = hf[patient]
                stored_p_feat = torch.tensor(patient_data['pathology_feature'][:])
                        
                similarity = F.cosine_similarity(
                    p_feat.flatten().unsqueeze(0),
                    stored_p_feat.flatten().unsqueeze(0)
                ).item()
                        
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gene = torch.tensor(patient_data['genomics_feature'][:])
        
        return best_gene.to('cuda')
    
class ProtoBank:
    def __init__(self, path='memory'):
        self.path = path
    
    def clear(self):       
        file_path = self.path
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    def save(self, memory_dict_list):
        
        with h5py.File(self.path, 'w') as hf:
            for i, patient_data in enumerate(memory_dict_list):
                group = hf.create_group(f"patient_{i}")
                for key in ['pathology_feature', 'pathology_prototype', 
                           'genomics_feature', 'genomics_prototype']:
                    data = patient_data[key]
                    if isinstance(data, torch.Tensor):
                        data = data.detach().cpu().numpy()
                    group.create_dataset(key, data=data)
    
    def retrievePathology(self, g_pro):
        best_similarity = -1
        best_pathology = None
        g_pro = g_pro.detach().cpu()
        if not isinstance(g_pro, torch.Tensor):
            g_pro = torch.tensor(g_pro)
        
        
        with h5py.File(self.path, 'r') as hf:
            for patient in hf:
                patient_data = hf[patient]
                stored_g_pro = torch.tensor(patient_data['genomics_prototype'][:])

                similarity = F.cosine_similarity(
                    g_pro.flatten().unsqueeze(0),
                    stored_g_pro.flatten().unsqueeze(0)
                    ).item()
                        
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pathology = torch.tensor(patient_data['pathology_feature'][:])
        
        return best_pathology.to('cuda')
    
    def retrieveGene(self, p_pro):

        best_similarity = -1
        best_gene = None
        p_pro = p_pro.detach().cpu()

        if not isinstance(p_pro, torch.Tensor):
            p_pro = torch.tensor(p_pro)
        
        with h5py.File(self.path, 'r') as hf:
            for patient in hf:
                patient_data = hf[patient]
                stored_p_pro = torch.tensor(patient_data['pathology_prototype'][:])
                        
                similarity = F.cosine_similarity(
                    p_pro.flatten().unsqueeze(0),
                    stored_p_pro.flatten().unsqueeze(0)
                    ).item()
                        
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gene = torch.tensor(patient_data['genomics_feature'][:])
        
        return best_gene.to('cuda')