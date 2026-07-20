import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import math

class NRE(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0.):
        
        super(NRE, self).__init__()

        self.ope_in_dim = in_dim[0]
        self.mas_in_dim = in_dim[1]
        self.pair_in_dim = in_dim[2]

        self.out_ope_dim = out_dim[0]
        self.out_mas_dim = out_dim[1]
        self.nega_slope = 0.2

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.ope_w = nn.Linear(self.ope_in_dim, self.out_ope_dim, bias=False)
        self.val_w = nn.Linear(self.ope_in_dim, self.out_ope_dim, bias=False)

        self.attn_src = nn.Parameter(torch.empty(size=(self.out_ope_dim, 1)))
        self.attn_dst = nn.Parameter(torch.empty(size=(self.out_ope_dim, 1)))
        
        if self.ope_in_dim != self.out_ope_dim:
            self.ope_res_fc = nn.Linear(self.ope_in_dim, self.out_ope_dim, bias=False)
        else:
            self.ope_res_fc = None

        self.d_ope_w = nn.Linear(self.out_ope_dim, self.out_mas_dim, bias=False)
        self.d_mas_w = nn.Linear(self.mas_in_dim, self.out_mas_dim, bias=False)
        self.d_pair_w = nn.Linear(self.pair_in_dim, self.out_mas_dim, bias=False)

        self.ope_alpha = nn.Parameter(torch.empty(size=(self.out_mas_dim, 1)))
        self.mas_alpha = nn.Parameter(torch.empty(size=(self.out_mas_dim, 1)))
        self.pair_alpha = nn.Parameter(torch.empty(size=(self.out_mas_dim, 1)))

        if self.mas_in_dim != self.out_mas_dim:
            self.mas_res_fc = nn.Linear(self.mas_in_dim, self.out_mas_dim, bias=False)
        else:
            self.mas_res_fc = None
        
        self.leaky_relu = nn.LeakyReLU(self.nega_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.nega_slope)

        nn.init.xavier_normal_(self.ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.val_w.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_src, gain=gain)
        nn.init.xavier_normal_(self.attn_dst, gain=gain)

        nn.init.xavier_normal_(self.d_ope_w.weight, gain=gain)
        nn.init.xavier_normal_(self.d_mas_w.weight, gain=gain)
        nn.init.xavier_normal_(self.d_pair_w.weight, gain=gain)

        nn.init.xavier_normal_(self.ope_alpha, gain=gain)
        nn.init.xavier_normal_(self.mas_alpha, gain=gain)
        nn.init.xavier_normal_(self.pair_alpha, gain=gain)

        if self.ope_res_fc is not None:
            nn.init.xavier_normal_(self.ope_res_fc.weight, gain=gain)

        if self.mas_res_fc is not None:
            nn.init.xavier_normal_(self.mas_res_fc.weight, gain=gain)
        
    def forward(self, opes_mask, proc_pair_adj, feats, jobs_gather=None):

        feat_ope = self.feat_drop(feats[0])
        feat_mas = self.feat_drop(feats[1])
        feat_pair = self.feat_drop(feats[2])

        h_ope = self.ope_w(feat_ope)
        v_ope = self.val_w(feat_ope)        

        attn_src = torch.matmul(h_ope, self.attn_src).squeeze(-1)
        attn_dst = torch.matmul(h_ope, self.attn_dst).squeeze(-1)
        e_ope = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2)
        e_ope = self.leaky_relu(e_ope) 
        
        ope_mask = (opes_mask == 1)
        e_ope = e_ope.masked_fill(~ope_mask, float('-9e10'))

        alpha_ope = F.softmax(e_ope, dim=-1)
        alpha_ope = self.attn_drop(alpha_ope)

        h_ope_agg = torch.matmul(alpha_ope, v_ope)

        if self.ope_res_fc is not None:
            h_ope_res = self.ope_res_fc(feat_ope)
        else:
            h_ope_res = feat_ope

        h_opes = h_ope_agg + h_ope_res
        h_jobs = h_opes.gather(1, jobs_gather)

        h_job = self.d_ope_w(h_jobs)
        h_mas = self.d_mas_w(feat_mas)
        h_pair = self.d_pair_w(feat_pair)

        attn_ope = torch.matmul(h_job, self.ope_alpha).squeeze(-1)
        attn_mas = torch.matmul(h_mas, self.mas_alpha).squeeze(-1)
        attn_pair = torch.matmul(h_pair, self.pair_alpha).squeeze(-1)

        e_pair =  attn_ope.unsqueeze(-1) + attn_mas.unsqueeze(-2) + attn_pair
        e_pair = self.leaky_relu(e_pair)

        mask_pair = (proc_pair_adj == 1)
        e_pair = e_pair.masked_fill(~mask_pair, float('-9e10'))

        alpha_pair = F.softmax(e_pair, dim=-2)
        alpha_pair = alpha_pair * mask_pair.float()
        alpha_pair = self.attn_drop(alpha_pair)

        h_mas_agg = torch.matmul(alpha_pair.transpose(1, 2), h_job)

        if self.mas_res_fc is not None:
            h_mas_res = self.mas_res_fc(feat_mas)
        else:
            h_mas_res = feat_mas

        h_mas = h_mas_agg + h_mas_res

        return h_opes, h_jobs, h_mas