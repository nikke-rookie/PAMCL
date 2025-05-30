import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss, bpr_loss_w, cl_loss, cross_cl_loss, InfoNCE
from util.logger import Log
from data.ui_graph import Interaction
from tqdm import tqdm
import time
import os
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Optional, Literal
from dataclasses import dataclass

#todo 测试 torch.jit 的加速效果
bpr_script = torch.jit.script(bpr_loss_w)
l2_reg_script = torch.jit.script(l2_reg_loss)
cl_script = torch.jit.script(cl_loss)


@dataclass
class Emb():
    user_embs: torch.Tensor  # 无意义, 仅作占位符tensor
    item_embs: torch.Tensor
    trained_weights: dict[str, torch.Tensor]

    user_embs_cl: torch.Tensor = torch.ones(1)
    item_embs_cl: torch.Tensor = torch.ones(1)

    user_pref_embs: Optional[torch.Tensor] = None

    image_embs: Optional[torch.Tensor] = None
    image_embs_cl: Optional[torch.Tensor] = None
    image_side_user: Optional[torch.Tensor] = None

    text_embs: Optional[torch.Tensor] = None
    text_embs_cl: Optional[torch.Tensor] = None
    text_side_user: Optional[torch.Tensor] = None


class PAMCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(PAMCL, self).__init__(conf, training_set, test_set, **kwargs)

        self.model_config = self.config['PAMCL']
        self.n_negs = int(self.model_config['n_negs'])
        self.temp = float(self.model_config['tau'])
        self.cl_rate = float(self.model_config['lambda'])
        self.device = torch.device(f"cuda:{int(self.config['gpu_id'])}" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
    
    def build(self):
        self.model = PAMCL_Encoder(self.data, self.emb_size, self.model_config, self.device, self.kwargs)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        self.scheduler = CosineAnnealingLR(optimizer, T_max=self.maxEpoch, eta_min=1e-4)
        train_start_time = time.time()
        start_batch100_time = time.time()
        for epoch in range(self.maxEpoch):
            for n, batch_data in enumerate(next_batch_pairwise(self.data, self.batch_size, self.n_negs)):
                user_ids, pos_ids, neg_ids = batch_data

                embs = model.forward(perturbed=True)
                rec_user_emb, rec_item_emb = embs.user_embs, embs.item_embs
                cl_user_emb, cl_item_emb = embs.user_embs_cl, embs.item_embs_cl
                image_embs = embs.image_embs
                text_embs = embs.text_embs
                user_pref_tensor = embs.user_pref_embs
                image_side_user, text_side_user = embs.image_side_user, embs.text_side_user

                trans_w = embs.trained_weights
                trans_w_list = [trans_w[key] for key in trans_w]

                user_emb, pos_item_emb, neg_item_embs = rec_user_emb[user_ids], rec_item_emb[pos_ids], rec_item_emb[neg_ids]

                # 用户偏好引导负样本采样
                if user_pref_tensor is not None:
                    #* 根据 neg_ids 取出对应中心性系数
                    item_id_centrality = self.data.item_id_centrality
                    neg_item_centralities = []
                    for neg_id in neg_ids:
                        neg_item_centralities.append([item_id_centrality[id] for id in neg_id])
                    
                    #* 负样本权重
                    neg_weights = torch.tensor(neg_item_centralities, dtype=torch.float, device=self.device)
                    weight_neg_item_embs: torch.Tensor = neg_weights.unsqueeze(-1) * neg_item_embs

                    user_pref: torch.Tensor = user_pref_tensor[user_ids]
                    # 计算相似度
                    user_pref = F.normalize(user_pref, p=2, dim=-1)
                    similarity = torch.bmm(weight_neg_item_embs, user_pref.unsqueeze(-1)).squeeze(-1)
                    sorted_indices = torch.argsort(similarity, descending=True, dim=-1)
                    lowest_sim_indices = sorted_indices[:, self.n_negs]
                    weight_neg_item_embs = neg_item_embs[torch.arange(len(neg_ids), device=self.device), lowest_sim_indices]

                    rec_loss1 = bpr_loss_w(user_emb, pos_item_emb, weight_neg_item_embs)
                else:
                    rec_loss1 = bpr_loss_w(user_emb, pos_item_emb, neg_item_embs)

                user_cl_loss = self.cl_rate * cl_script(user_ids, rec_user_emb, cl_user_emb, self.temp, self.device) # type: ignore
                item_cl_loss = self.cl_rate * cl_script(pos_ids, rec_item_emb, cl_item_emb, self.temp, self.device) # type: ignore
                ui_cl_loss = user_cl_loss + item_cl_loss

                ccl_loss = 0.
                # if self.data.image_modal and self.data.text_modal:
                #     u_loss = self.cl_rate*cl_loss(user_ids, rec_user_emb+image_side_user, rec_user_emb+text_side_user, self.temp, self.device)
                #     i_loss = self.cl_rate*cl_loss(pos_ids, rec_item_emb+image_embs, rec_item_emb+text_embs, self.temp, self.device)
                #     ccl_loss = u_loss + i_loss

                # total_cl_loss = ui_cl_loss + image_cl_loss + text_cl_loss
                
                if image_embs is not None and text_embs is not None:
                    l2_loss = l2_reg_loss(self.reg, [user_emb, pos_item_emb, image_embs[pos_ids], text_embs[pos_ids]], self.device)
                    l2_loss += l2_reg_loss(self.reg, trans_w_list, self.device)
                elif image_embs is not None:
                    l2_loss = l2_reg_loss(self.reg, [user_emb, pos_item_emb, image_embs[pos_ids]], self.device)
                elif text_embs is not None:
                    l2_loss = l2_reg_loss(self.reg, [user_emb, pos_item_emb, text_embs[pos_ids]], self.device)
                else:
                    l2_loss = l2_reg_loss(self.reg, [user_emb, pos_item_emb], self.device)

                batch_loss = rec_loss1 + l2_loss + ui_cl_loss # type: ignore
            
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    end_batch100_time = time.time()
                    elapsed_time = end_batch100_time - start_batch100_time
                    start_batch100_time = time.time()
                    self.model_log.add(f"epoch: {epoch+1}, batch: {n}, time: {elapsed_time:.4f}s, rec_loss: {rec_loss1.item()}, cl_loss: {ui_cl_loss.item()}") # type: ignore

            with torch.no_grad():
                embs = self.model.forward()
                self.user_emb, self.item_emb = embs.user_embs, embs.item_embs
            self.fast_evaluation(epoch)
            self.scheduler.step()
            
            if self.early_stop == 20:
                break

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        self.best_user_emb, self.best_item_emb = self.user_emb, self.item_emb

    def persist(self):
        user_embs = self.best_user_emb.cpu().numpy()
        item_embs = self.best_item_emb.cpu().numpy()
        embs = {'user': user_embs, 'item': item_embs}
        save_name = f"{self.config['model']['name']}_{self.timestamp}.safetensors"
        save_file(embs, save_name)

    def predict(self, u):
        user_id = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class PAMCL_Encoder(nn.Module):
    def __init__(self, data: Interaction, emb_size: int, model_config: dict, device: torch.device, kwargs):
        super(PAMCL_Encoder, self).__init__()

        self.data = data
        self.emb_size = emb_size
        self.device = device
        self.model_name = kwargs.get('model_name')
        self.timestamp = kwargs.get('timestamp')

        self.eta = model_config['eta']
        self.n_layer = model_config['n_layer']
        self.cl_layer = model_config['cl_layer']

        self.norm_adj = self.data.norm_adj
        self.param_dict = self._init_model()
        self.image_modal_flag, self.text_modal_flag, self.pref_flag = False, False, False
        self._init_multi_modal()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj, device=self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        param_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size, device=self.device))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size, device=self.device))),
            
            'u_w_q': nn.Parameter(initializer(torch.empty([self.emb_size, self.emb_size], device=self.device))),
            'u_w_k': nn.Parameter(initializer(torch.empty([self.emb_size, self.emb_size], device=self.device))),
            'u_w_v': nn.Parameter(initializer(torch.empty([self.emb_size, self.emb_size], device=self.device))),

            'i_w_q': nn.Parameter(initializer(torch.empty([self.emb_size, self.emb_size], device=self.device))),
            'i_w_k': nn.Parameter(initializer(torch.empty([self.emb_size, self.emb_size], device=self.device))),
            'i_w_v': nn.Parameter(initializer(torch.empty([self.emb_size, self.emb_size], device=self.device))),
        })
        return param_dict


    def _init_multi_modal(self):
        image_modal = self.data.image_modal
        text_modal = self.data.text_modal
        user_pref = self.data.user_pref

        if image_modal:
            Log.cli('Model', f'📷 Loading image safetensors to {self.device} and project to {self.emb_size} dimensions')
            
            image_projection = nn.Linear(int(image_modal['dim']), self.emb_size, device=self.device)
            if image_modal['pre_trained']['enable']:
                try:
                    image_pth = image_modal['pre_trained']['image_pth']
                    image_projection.load_state_dict(torch.load(image_pth))
                except Exception as e:
                    Log.catch(e, 'image_modal', '_init_multi_modal')
                    exit(-1)
            else:
                if image_modal['pre_trained']['save']:
                    path = image_modal['pre_trained']['save_path']
                    os.makedirs(f"{path}/{self.model_name}_{self.timestamp}", exist_ok=True)
                    torch.save(image_projection.state_dict(), f'{path}/{self.model_name}_{self.timestamp}/image.pth')
            
            origin_image_tensor = torch.empty(size=(self.data.item_num, int(image_modal['dim'])), device=self.device)

            if str(image_modal['image_set']).endswith('npy'):
                origin_image_np = np.load(image_modal['image_set'])
                origin_image_tensor = torch.from_numpy(origin_image_np).to(self.device, dtype=torch.float32)
            else:
                with safe_open(image_modal['image_set'], 'pt', device=f"cuda:{self.device.index}") as f: # type: ignore
                    for idx, item in tqdm(enumerate(self.data.item), desc='item image'):
                        origin_image_tensor[idx] = f.get_tensor(item)
            self.param_dict['image_embs_tensor'] = image_projection(origin_image_tensor)
            self.image_modal_flag = True

        if text_modal:
            Log.cli('Model', f'📒 Loading text safetensors to {self.device} and project to {self.emb_size} dimensions')
            item_text_projection = nn.Linear(int(text_modal['dim']), self.emb_size, device=self.device)
            if text_modal['pre_trained']['enable']:
                try:
                    item_text_pth = text_modal['pre_trained']['item_text_pth']
                    item_text_projection.load_state_dict(torch.load(item_text_pth))
                except Exception as e:
                    Log.catch(e, 'text_modal', '_init_multi_modal')
                    exit(-1)
            else:
                if text_modal['pre_trained']['save']:
                    path = text_modal['pre_trained']['save_path']
                    os.makedirs(f"{path}/{self.model_name}_{self.timestamp}", exist_ok=True)
                    torch.save(item_text_projection.state_dict(), f'{path}/{self.model_name}_{self.timestamp}/item_text.pth')

            origin_text_tensor = torch.empty(size=(self.data.item_num, int(text_modal['dim'])), device=self.device)

            if str(text_modal['item_text']).endswith('npy'):
                origin_text_np = np.load(text_modal['item_text'])
                origin_text_tensor = torch.from_numpy(origin_text_np).to(self.device, dtype=torch.float32)
            else:
                with safe_open(text_modal['item_text'], 'pt', device=f"cuda:{self.device.index}") as f1: # type: ignore
                    for idx, item in tqdm(enumerate(self.data.item), desc='item text'):
                        origin_text_tensor[idx] = f1.get_tensor(item)
            
            self.param_dict['item_text_tensor'] = item_text_projection(origin_text_tensor)
            self.text_modal_flag = True
        
        if user_pref:
            Log.cli('Model', f'📒 Loading pref safetensors to {self.device} and project to {self.emb_size} dimensions')
            user_pref_projection = nn.Linear(int(user_pref['dim']), self.emb_size, device=self.device)
            if user_pref['pre_trained']['enable']:
                try:
                    user_pref_pth = user_pref['pre_trained']['user_pref_pth']
                    user_pref_projection.load_state_dict(torch.load(user_pref_pth))
                except Exception as e:
                    Log.catch(e, 'user_pref', '_init_multi_modal')
                    exit(-1) 
            else:
                if user_pref['pre_trained']['save']:
                    path = user_pref['pre_trained']['save_path']
                    os.makedirs(f"{path}/{self.model_name}_{self.timestamp}", exist_ok=True)
                    torch.save(user_pref_projection.state_dict(), f'{path}/{self.model_name}_{self.timestamp}/user_pref.pth')
            
            origin_pref_tensor = torch.empty(size=(self.data.user_num, int(user_pref['dim'])), device=self.device)
            with safe_open(self.data.user_pref['pref_embs'], 'pt', device=f"cuda:{self.device.index}") as f2: # type: ignore
                for idx, user in tqdm(enumerate(self.data.user), desc='user pref'):
                    origin_pref_tensor[idx] = f2.get_tensor(user)
            
            self.param_dict['user_pref'] = user_pref_projection(origin_pref_tensor)
            self.pref_flag = True

    def SelfAttention(self, trans_w, emb_1: torch.Tensor, emb_2: torch.Tensor, emb_3: torch.Tensor, mode: Literal['u','i']):  
        q = emb_1.unsqueeze(1)
        k = emb_2.unsqueeze(1)
        v = emb_3.unsqueeze(1)
        
        if mode == 'u':
            Q = torch.matmul(q, trans_w['u_w_q'])
            K = torch.matmul(k, trans_w['u_w_k'])
            V = torch.matmul(v, trans_w['u_w_v'])
        elif mode == 'i':
            Q = torch.matmul(q, trans_w['i_w_q'])
            K = torch.matmul(k, trans_w['i_w_k'])
            V = torch.matmul(v, trans_w['i_w_v'])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
        att = F.softmax(scores, dim=-1)

        Z = torch.matmul(att, V).squeeze(1)
        Z = F.normalize(Z, p=2, dim=-1)
        return Z

    
    def forward(self, perturbed=False):
        final_image_embeddings, final_text_embeddings = None, None
        
        embs = Emb(
            user_embs = self.param_dict['user_emb'],
            item_embs = self.param_dict['item_emb'],
            # user_pref_embs = self.param_dict['user_pref'],
            trained_weights = {
                'u_w_q': self.param_dict['u_w_q'],
                'u_w_k': self.param_dict['u_w_k'],
                'u_w_v': self.param_dict['u_w_v'],
                'i_w_q': self.param_dict['i_w_q'],
                'i_w_k': self.param_dict['i_w_k'],
                'i_w_v': self.param_dict['i_w_v'],
            }
        )

        if self.image_modal_flag:
            image_side_embs = torch.cat([embs.user_embs, self.param_dict['image_embs_tensor']], 0)
            all_image_embeddings = []
            for k in range(self.n_layer):
                image_side_embs = torch.sparse.mm(self.sparse_norm_adj, image_side_embs)
                all_image_embeddings.append(image_side_embs)

            final_image_embeddings = torch.mean(torch.stack(all_image_embeddings, dim=1), dim=1)
            final_image_embeddings = F.leaky_relu(final_image_embeddings)
            final_image_embeddings = nn.Dropout(p=0.2)(final_image_embeddings)
            final_image_embeddings = F.normalize(final_image_embeddings, p=2)
        
        if self.text_modal_flag:
            text_side_embs = torch.cat([embs.user_embs, self.param_dict['item_text_tensor']], 0)
            all_text_embeddings = []
            for k in range(self.n_layer):
                text_side_embs = torch.sparse.mm(self.sparse_norm_adj, text_side_embs)
                all_text_embeddings.append(text_side_embs)
            
            final_text_embeddings = torch.mean(torch.stack(all_text_embeddings, dim=1), dim=1)
            final_text_embeddings = F.leaky_relu(final_text_embeddings)
            final_text_embeddings = nn.Dropout(p=0.2)(final_text_embeddings)
            final_text_embeddings = F.normalize(final_text_embeddings, p=2)
        
        if final_image_embeddings is not None and final_text_embeddings is not None:
            embs.image_side_user, embs.image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
            embs.text_side_user, embs.text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])
            
            rate = 0.5
            attn_user = self.SelfAttention(embs.trained_weights, embs.user_embs, embs.image_side_user, embs.text_side_user, mode='u')
            fusion_user_embeddings = embs.user_embs + rate*attn_user

            attn_item = self.SelfAttention(embs.trained_weights, embs.item_embs, embs.image_embs, embs.text_embs, mode='i')
            fusion_item_embeddings = embs.item_embs + rate*attn_item

            joint_embeddings = torch.cat([fusion_user_embeddings, fusion_item_embeddings], dim=0)

            # embs.image_side_user, embs.image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
            # embs.text_side_user, embs.text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])
            # fusion_user_embeddings = torch.mean(torch.stack([self.param_dict['user_emb'], embs.image_side_user], dim=0), dim=0)
            # fusion_item_embeddings = torch.mean(torch.stack([self.param_dict['item_emb'], embs.text_embs], dim=0), dim=0)
            # joint_embeddings = torch.cat([fusion_user_embeddings, fusion_item_embeddings], dim=0)

        else:
            # print('No multi-modal')
            joint_embeddings = torch.cat([embs.user_embs, embs.item_embs], 0)

        all_embeddings = []
        all_embeddings_cl = joint_embeddings

        for k in range(self.n_layer):
            joint_embeddings = torch.sparse.mm(self.sparse_norm_adj, joint_embeddings)
            if perturbed:
                random_noise = torch.rand_like(joint_embeddings)
                joint_embeddings += torch.sign(joint_embeddings) * F.normalize(random_noise, dim=-1) * self.eta

            all_embeddings.append(joint_embeddings)

            if k == self.cl_layer-1:
                all_embeddings_cl = joint_embeddings

        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        embs.user_embs, embs.item_embs = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        embs.user_embs_cl, embs.item_embs_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        
        return embs
