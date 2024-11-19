import torch
import torch.nn as nn
import torch.nn.functional as F

from gratif.attention import MAB2


class Encoder(nn.Module):
    def __init__(self, dim=41, nkernel=128, n_layers=3, attn_head=4, device="cuda", wocat=False):
        super(Encoder, self).__init__()
        self.dim = dim + 2
        self.nheads = attn_head
        self.nkernel = nkernel
        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)
        self.n_layers = n_layers
        self.channel_time_attn = nn.ModuleList()
        self.time_channel_attn = nn.ModuleList()
        self.edge_nn = nn.ModuleList()
        self.channel_attn = nn.ModuleList()
        self.device = device
        self.wocat = wocat
        self.output = nn.Linear(3 * nkernel, 1)
        for i in range(self.n_layers):
            self.channel_time_attn.append(MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, self.nheads))
            self.time_channel_attn.append(MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, self.nheads))
            self.edge_nn.append(nn.Linear(3 * nkernel, nkernel))
            self.channel_attn.append(MAB2(nkernel, nkernel, nkernel, nkernel, self.nheads))
        self.relu = nn.ReLU()

    def gather(self, x, inds):
        # inds =  # keep repeating until the embedding len as a new dim

        # print("#"*50)
        # print(inds[:, :, None].repeat(1, 1, x.shape[-1]).size())
        # print(inds.size())
        # print(x.size())
        # print(x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1])).size())
        # print("#"*50)

        return x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1]))

    def forward(self, context_x, value, mask, target_value, target_mask):

        """
        context_x (B, T, C)
        value (B, T, C)
        mask (B, T, C)
        target_value (B, T, C)
        target_mask (B, T, C)

        =>
        K_ = max(TxC)
        output (B, K_, 1)
        target_U_ (B, k_)
        target_mask_ (B, k_)
        """

        seq_len = context_x.size(-1)  # T = length of time series
        ndims = value.shape[-1]  # C = # of channels
        # print(f"seq_len = {seq_len}, ndims = {ndims}")
        T = context_x[:, :, None]  # BxTx1
        C = torch.ones([context_x.shape[0], ndims]).cumsum(1).to(self.device) - 1  # BxC intialization for one hot encoding channels => 각 채널에 0,1,2,... 와 같은 인덱스 생성
        # print(f"T size = {T.size()}, C size = {C.size()}")
        T_inds = torch.cumsum(torch.ones_like(value).to(torch.int64), 1) - 1  # BxTxC init for time indices => timepoint의 인덱스 
        C_inds = torch.cumsum(torch.ones_like(value).to(torch.int64), -1) - 1  # BxTxC init for channel indices => channel의 인덱스
        mk_bool = mask.to(torch.bool)  # BxTxC
        full_len = torch.max(mask.sum((1, 2))).to(torch.int64)  # flattened TxC max length possible => 각 배치에서 mask의 모든 값을 더해 나온 최대 길이 = 가장 긴 시계열의 길이
        # print(f"full length = {full_len}")
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0) # 주어진 텐서 v에 대해, 필요한 길이만큼 0으로 패딩 => full_len으로 맞추기

        # flattening to 2D
        ## r[m] for r, m in zip(T_inds, mk_bool) => for each batch, mk_bool==1 인 r의 값들만 가져옴. => pad로 full_len 맞춰주기.
        T_inds_ = torch.stack([pad(r[m]) for r, m in zip(T_inds, mk_bool)]).contiguous()  # BxTxC -> Bxfull_len
        U_ = torch.stack([pad(r[m]) for r, m in zip(value, mk_bool)]).contiguous()  # BxTxC (values) -> Bxfull_len // observation의 edge feature = observation value
        target_U_ = torch.stack([pad(r[m]) for r, m in zip(target_value, mk_bool)]).contiguous()  # BxK_ = Bxfull_len // tagert edge = target value = 예측하고자 하는 값
        target_mask_ = torch.stack([pad(r[m]) for r, m in zip(target_mask, mk_bool)]).contiguous()  # BxK_ = Bxfull_len // target_mask에서 mask==1인 값들만 가져옴. 나중에 output masking 할때 사용 
                                                                                                    # encoder에 들어오기 전에 context_mask=obs mask+target mask해둬서 mk_bool에도 반영됨. 이렇게 구해도 target mask만 나옴.
        C_inds_ = torch.stack([pad(r[m]) for r, m in zip(C_inds, mk_bool)]).contiguous()  # BxK_ = Bxfull_len
        mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous()  # BxK_ = Bxfull_len
        # print(T_inds_.size(), U_.size(), target_U_.size(), target_mask_.size(), C_inds_.size(), mk_.size())

        obs_len = full_len

        C_ = torch.nn.functional.one_hot(C.to(torch.int64), num_classes=ndims).to(torch.float32)  # BxCxC #channel one hot encoding
        U_indicator = 1 - mk_ + target_mask_ # target의 edge indicator
        # print(f"before concate, U_ size = {U_.size()} and U_indicator size = {U_indicator.size()}")
        U_ = torch.cat([U_[:, :, None], U_indicator[:, :, None]], -1)  # BxK_max x 2 #todo: correct
        # print(f"after concate, U_ size = {U_.size()}")

        # creating Channel mask and Time mask
        C_mask = C[:, :, None].repeat(1, 1, obs_len) # C는 BxC => BxCxK_
        temp_c_inds = C_inds_[:, None, :].repeat(1, ndims, 1) # C_inds_는 BxK_ => Bx ndims xK_ = BxCxK_
        C_mask = (C_mask == temp_c_inds).to(torch.float32)  # BxCxK_
        C_mask = C_mask * mk_[:, None, :].repeat(1, C_mask.shape[1], 1) # C_mask에서 유효한 위치만을 1로 남겨둬, mk_가 True인 위치만 어텐션에 참여하도록.

        T_mask = T_inds_[:, None, :].repeat(1, T.shape[1], 1)
        temp_T_inds = torch.ones_like(T[:, :, 0]).cumsum(1)[:, :, None].repeat(1, 1, C_inds_.shape[1]) - 1
        T_mask = (T_mask == temp_T_inds).to(torch.float32)  # BxTxK_
        T_mask = T_mask * mk_[:, None, :].repeat(1, T_mask.shape[1], 1)

        # print(f"before embedding: edge size = {U_.size()}, time node size = {T.size()}, channel node size = {C_.size()}")
        U_ = self.relu(self.edge_init(U_)) * mk_[:, :, None].repeat(1, 1, self.nkernel)  # edge embedding
        T_ = torch.sin(self.time_init(T))  # learned time node embedding
        C_ = self.relu(self.chan_init(C_))  # one-hot channel node embedding
        # print(f"after embedding: edge embedding size = {U_.size()}, time node embeeding size = {T_.size()}, channel node embedding size = {C_.size()}")

        del temp_T_inds
        del temp_c_inds

        for i in range(self.n_layers):
            # channels as queries
            q_c = C_ # channel node embedding as queries
            k_t = self.gather(T_, T_inds_)  # BxK_max x embd_len, neighbor(connected) timepoints node embedding
                                            # 'T_inds_' 가 정의된 걸 보면, timepoints index인 'T_inds'에서 mk_bool==1을 만족하는 index만 가져오니까, 각 channel이랑 연결된 timepoints의 embedding만 얻을 수 있음.
                                            # gather함수를 사용해서 T_에 정의된 embedding들 중에서, T_inds_에 필요한 애들을 차원에 맞게 중복으로 다 가져올 수 있음.
            k = torch.cat([k_t, U_], -1)  # BxK_max x 2 * embd_len, [neighbor(connected) timepoints node embedding || edge embedding] as keys & values
            # print(f"query size = {q_c.size()}, neighbor node size = {k_t.size()}, key&value size = {k.size()}")

            C__ = self.channel_time_attn[i](q_c, k, C_mask)  # attn (channel_embd, concat(time, values)) along with the mask

            # times as queries
            q_t = T_
            k_c = self.gather(C_, C_inds_)
            k = torch.cat([k_c, U_], -1)
            T__ = self.time_channel_attn[i](q_t, k, T_mask)

            # updating edge weights
            U_ = self.relu(U_ + self.edge_nn[i](torch.cat([U_, k_t, k_c], -1))) * mk_[:, :, None].repeat(1, 1, self.nkernel)

            # updating only channel nodes
            if self.wocat:
                C_ = C__
            else:
                C_ = self.channel_attn[i](C__, C__)

            T_ = T__

        k_t = self.gather(T_, T_inds_)
        k_c = self.gather(C_, C_inds_)
        output = self.output(torch.cat([U_, k_t, k_c], -1))
        # print("#"*50)
        # print(f"U_ size = {U_.size()}, k_t size = {k_t.size()}, k_c size = {k_c.size()}, concat size = {torch.cat([U_, k_t, k_c], -1).size()}")
        # print(f"output size = {output.size()}, target_U_ size = {target_U_.size()}, target_mask_ size = {target_mask_.size()}")

        return output, target_U_, target_mask_
