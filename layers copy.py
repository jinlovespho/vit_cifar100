import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


# authors attention
class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out



# nn_mha_attention
# class TransformerEncoder(nn.Module):
#     def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
#         super(TransformerEncoder, self).__init__()
#         self.la1 = nn.LayerNorm(feats)
#         self.msa = nn.MultiheadAttention(embed_dim=feats, num_heads=head, dropout=dropout)
#         # self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
#         self.la2 = nn.LayerNorm(feats)
#         self.mlp = nn.Sequential(
#             nn.Linear(feats, mlp_hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden, feats),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
        
        
#     def forward(self, x):
#         x = x.transpose(0, 1)  # nn.MultiheadAttention expects input as (seq_len, batch, features)
#         attn_output, _ = self.msa(x, x, x)  # query, key, value are all the same for self-attention
#         out = attn_output.transpose(0, 1)  # transpose back to original shape
#         out = self.la1(out) + x.transpose(0, 1)  # Residual connection
#         out = self.mlp(self.la2(out)) + out  # Another layer and residual connection
#         return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.head_dim = self.feats // self.head 

        assert self.head_dim == self.feats // self.head 

        
        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        
        # lora ? 
        # self.q_A = nn.Linear(feats, r)
        # self.q_B = nn.Linear(r, feats)
               
        # token-wise attn 
        # non overlapping head layers for 12heads.
        self.h0_linear = nn.Linear(feats, feats)
        self.h1_linear = nn.Linear(feats, feats)
        self.h2_linear = nn.Linear(feats, feats)
        self.h3_linear = nn.Linear(feats, feats)
        self.h4_linear = nn.Linear(feats, feats)
        self.h5_linear = nn.Linear(feats, feats)
        self.h6_linear = nn.Linear(feats, feats)
        self.h7_linear = nn.Linear(feats, feats)
        self.h8_linear = nn.Linear(feats, feats)
        self.h9_linear = nn.Linear(feats, feats)
        self.h10_linear = nn.Linear(feats, feats)
        self.h11_linear = nn.Linear(feats, feats)
        
        self.out_linear = nn.Linear(int(feats/3), feats)
        
        
        # dim wise attn
        self.h0_dim = nn.Linear(feats//self.head, feats)
        self.h1_dim = nn.Linear(feats//self.head, feats)
        self.h2_dim = nn.Linear(feats//self.head, feats)
        self.h3_dim = nn.Linear(feats//self.head, feats)
        self.h4_dim = nn.Linear(feats//self.head, feats)
        self.h5_dim = nn.Linear(feats//self.head, feats)
        self.h6_dim = nn.Linear(feats//self.head, feats)
        self.h7_dim = nn.Linear(feats//self.head, feats)
        self.h8_dim = nn.Linear(feats//self.head, feats)
        self.h9_dim = nn.Linear(feats//self.head, feats)
        self.h10_dim = nn.Linear(feats//self.head, feats)
        self.h11_dim = nn.Linear(feats//self.head, feats)
        
        self.out_dim = nn.Linear(feats//3*12, feats)
        
        # 3dk
        self.h0_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h1_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h2_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h3_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h4_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h5_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h6_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h7_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h8_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h9_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h10_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        self.h11_dim_3dk = nn.Linear(feats//self.head, 3*feats//self.head)
        
        self.out_dim_3dk = nn.Linear(feats, feats)
        

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # 택 1
        # o = self.vanilla_attention(x)
        # o = self.non_overlapping_head_attention_token_wise(x)
        # o = self.non_overlapping_head_attention_dimension_wise(x)
        o = self.non_overlapping_head_attention_dimension_wise_3dk(x)
        
        return o
    
    def vanilla_attention(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)       # self.q(x) 의 output에 .view
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)       # 최종: 현재 q,k,v shape: (batch_size, head, seq_len, head_dim)
                                                                                        #                        (b, h, n, head_dim)
        
        # attention 계산 part1. softmax(QK/d)
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1)   #(b, h, n, n)
        
        # attention 계산 part2. softmax(QK/d)*V 
        attn = torch.einsum('bhij, bhjk -> bhik', score, v) # (b, h, n, f//h)
        attn = attn.permute(0, 2, 1, 3) # (b, n, h, f//h )
        
        # attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b, n, h, f//h)
        
        # output projection layer 계산 O
        tmp = attn.flatten(start_dim=2)           # start_dim=2 이면 attn:(b, n, h, f//h ) 에서 h 부터 뒤에를 다 flatten 시키는 것. => (b, n, h * f//h ) = (b, n, f)
        tmp = self.o(tmp)
        o = self.dropout(tmp)
        # o = self.dropout(self.o(attn.flatten(2)))
    
        
        return o
        
    def non_overlapping_head_attention_token_wise(self, x):                             # x : (b,n,f) = (1024, 65, 384
        # breakpoint()
        b, n, d = x.size()
        num_head = self.head 
        
        # 일단 12 개의 head가 (head1, head2, , , head12) input x = (n, d) 에서 n 을 중복되지 않고 볼려면, 
        # 먼저 input x = (b, n, d) 를 12개의 chunk로 나누어야. along dim=1 (n).
        # but 문제가 tmp = x.chunk(12, dim=1) 로 하면 tmp가 11개 나온다. 12 개 나와야 되는데.
        # 그래서 일단 x를 12개로 나누고 각각 12개의 linear layer들로 태워서 q,k,v로 나누어 보자 !
        
        # x: (1024, 65, 384)
        x_cls = x[:,0:1,:]
        x = x[:,1:,:]
        x_chunked = list(x.chunk(13, dim=1))        # x.chunk return value 는 set() 꼴이라 아래 torch.cat을 허용 X. 따라서 list로 변환
        
        x_chunked[0] = torch.cat( [x_cls, x_chunked[0], x_chunked[12][:,0:1,:]], dim=1 )
        x_chunked[1] = torch.cat( [x_cls, x_chunked[1], x_chunked[12][:,1:2,:]], dim=1 )
        x_chunked[2] = torch.cat( [x_cls, x_chunked[2], x_chunked[12][:,2:3,:]], dim=1 )
        x_chunked[3] = torch.cat( [x_cls, x_chunked[3], x_chunked[12][:,3:4,:]], dim=1 )
        
        x_chunked[4] = torch.cat( [x_cls, x_chunked[4]], dim=1 )
        x_chunked[5] = torch.cat( [x_cls, x_chunked[5]], dim=1 )
        x_chunked[6] = torch.cat( [x_cls, x_chunked[6]], dim=1 )
        x_chunked[7] = torch.cat( [x_cls, x_chunked[7]], dim=1 )
        x_chunked[8] = torch.cat( [x_cls, x_chunked[8]], dim=1 )
        x_chunked[9] = torch.cat( [x_cls, x_chunked[9]], dim=1 )
        x_chunked[10] = torch.cat( [x_cls, x_chunked[10]], dim=1 )
        x_chunked[11] = torch.cat( [x_cls, x_chunked[11]], dim=1 )
        
        del x_chunked[12]
        
        # 총 12개의 head로 나누기 위해 input data x 를 12 부분으로 나누었다.
        # x_chunked[0], ... , x_chunked[3] => shape:(1024, 1+5+1, 384) = (bs, cls+5+leftover, hidden_dim)
        # x_chunked[4], ... , x_chunked[11] => shape: (1024, 1+5, 384) = (bs, cls+5, hidden+dim)
        
        # 생각정리
        # cls token 도 각 head에 독립적으로 cls0, cls1, , , cls11붙여줄지
        # 또는 동일한 하나의 cls token을 모든 head에 붙여줄지. <- 이게 맞지. cls token은 모든 정보를 봐야 O
        
        # 이미 모든 x_chunked[i] 에는 가장 위에 동일한 cls_token이 포함되어 있다.
        h0_qkv = self.h0_linear(x_chunked[0])
        h1_qkv = self.h1_linear(x_chunked[1])
        h2_qkv = self.h2_linear(x_chunked[2])
        h3_qkv = self.h3_linear(x_chunked[3])
        h4_qkv = self.h4_linear(x_chunked[4])
        h5_qkv = self.h5_linear(x_chunked[5])
        h6_qkv = self.h6_linear(x_chunked[6])
        h7_qkv = self.h7_linear(x_chunked[7])
        h8_qkv = self.h8_linear(x_chunked[8])
        h9_qkv = self.h9_linear(x_chunked[9])
        h10_qkv = self.h10_linear(x_chunked[10])
        h11_qkv = self.h11_linear(x_chunked[11])

        h0_qkv_chunk = list(h0_qkv.chunk(3, dim=2))
        h1_qkv_chunk = list(h1_qkv.chunk(3, dim=2))
        h2_qkv_chunk = list(h2_qkv.chunk(3, dim=2))
        h3_qkv_chunk = list(h3_qkv.chunk(3, dim=2))
        h4_qkv_chunk = list(h4_qkv.chunk(3, dim=2))
        h5_qkv_chunk = list(h5_qkv.chunk(3, dim=2))
        h6_qkv_chunk = list(h6_qkv.chunk(3, dim=2))
        h7_qkv_chunk = list(h7_qkv.chunk(3, dim=2))
        h8_qkv_chunk = list(h8_qkv.chunk(3, dim=2))
        h9_qkv_chunk = list(h9_qkv.chunk(3, dim=2))
        h10_qkv_chunk = list(h10_qkv.chunk(3, dim=2))
        h11_qkv_chunk = list(h11_qkv.chunk(3, dim=2))
        
        h0_q = h0_qkv_chunk[0]
        h0_k = h0_qkv_chunk[1]
        h0_v = h0_qkv_chunk[2]  # (1024, 7, 128)=(b,n,hd)
        
        # breakpoint()
        
        score0 = F.softmax(torch.einsum('bij, bkj -> bik', h0_q, h0_k)/self.sqrt_d, dim=-1)   # (b,n,n)
        attn0 = torch.einsum('bij, bjk -> bik', score0, h0_v)   # (b,n,hd)
        
        score1 = F.softmax(torch.einsum('bij, bkj -> bik', h1_qkv_chunk[0], h1_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn1 = torch.einsum('bij, bjk -> bik', score1, h1_qkv_chunk[2])   # (b,n,hd)
        
        score2 = F.softmax(torch.einsum('bij, bkj -> bik', h2_qkv_chunk[0], h2_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn2 = torch.einsum('bij, bjk -> bik', score2, h2_qkv_chunk[2])   # (b,n,hd)
        
        score3 = F.softmax(torch.einsum('bij, bkj -> bik', h3_qkv_chunk[0], h3_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn3 = torch.einsum('bij, bjk -> bik', score3, h3_qkv_chunk[2])   # (b,n,hd)
        
        score4 = F.softmax(torch.einsum('bij, bkj -> bik', h4_qkv_chunk[0], h4_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn4 = torch.einsum('bij, bjk -> bik', score4, h4_qkv_chunk[2])   # (b,n,hd)
        
        score5 = F.softmax(torch.einsum('bij, bkj -> bik', h5_qkv_chunk[0], h5_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn5 = torch.einsum('bij, bjk -> bik', score5, h5_qkv_chunk[2])   # (b,n,hd)

        score6 = F.softmax(torch.einsum('bij, bkj -> bik', h6_qkv_chunk[0], h6_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn6 = torch.einsum('bij, bjk -> bik', score6, h6_qkv_chunk[2])   # (b,n,hd)
        
        score7 = F.softmax(torch.einsum('bij, bkj -> bik', h7_qkv_chunk[0], h7_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn7 = torch.einsum('bij, bjk -> bik', score7, h7_qkv_chunk[2])   # (b,n,hd)
        
        score8 = F.softmax(torch.einsum('bij, bkj -> bik', h8_qkv_chunk[0], h8_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn8 = torch.einsum('bij, bjk -> bik', score8, h8_qkv_chunk[2])   # (b,n,hd)

        score9 = F.softmax(torch.einsum('bij, bkj -> bik', h9_qkv_chunk[0], h9_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn9 = torch.einsum('bij, bjk -> bik', score9, h9_qkv_chunk[2])   # (b,n,hd)
        
        score10 = F.softmax(torch.einsum('bij, bkj -> bik', h10_qkv_chunk[0], h10_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn10 = torch.einsum('bij, bjk -> bik', score10, h10_qkv_chunk[2])   # (b,n,hd)
        
        score11 = F.softmax(torch.einsum('bij, bkj -> bik', h11_qkv_chunk[0], h11_qkv_chunk[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn11 = torch.einsum('bij, bjk -> bik', score11, h11_qkv_chunk[2])   # (b,n,hd)
        
        # breakpoint()
        
        cls_attn0 = attn0[:,0:1,:]
        cls_attn1 = attn1[:,0:1,:]
        cls_attn2 = attn2[:,0:1,:]
        cls_attn3 = attn3[:,0:1,:]
        cls_attn4 = attn4[:,0:1,:]
        cls_attn5 = attn5[:,0:1,:]
        cls_attn6 = attn6[:,0:1,:]
        cls_attn7 = attn7[:,0:1,:]
        cls_attn8 = attn8[:,0:1,:]
        cls_attn9 = attn9[:,0:1,:]
        cls_attn10 = attn10[:,0:1,:]
        cls_attn11 = attn11[:,0:1,:]    # (b, 1, hd)
        
        all_cls = (cls_attn0 + cls_attn1 + cls_attn2 + cls_attn3 + cls_attn4 + cls_attn5 + 
                   cls_attn6 + cls_attn7 + cls_attn8 + cls_attn9 + cls_attn10 + cls_attn11 )/12
        
        
        attn0 = attn0[:,1:,:]
        attn1 = attn1[:,1:,:]
        attn2 = attn2[:,1:,:]
        attn3 = attn3[:,1:,:]
        attn4 = attn4[:,1:,:]
        attn5 = attn5[:,1:,:]
        attn6 = attn6[:,1:,:]
        attn7 = attn7[:,1:,:]
        attn8 = attn8[:,1:,:]
        attn9 = attn9[:,1:,:]
        attn10 = attn10[:,1:,:]                                               
        attn11 = attn11[:,1:,:]     # (b, n, hd)
        
        # breakpoint()
        
        all_attn = torch.cat([attn0, attn1, attn2, attn3, attn4, attn5,
                              attn6, attn7, attn8, attn9, attn10, attn11], dim=1)
        
        final_cls_all_attn = torch.cat([all_cls, all_attn], dim=1)  # (b, n, hd)
        
        out = self.out_linear(final_cls_all_attn)
        out = self.dropout(out)
        
        # breakpoint()
        
        return out
        

    # def non_overlapping_head_attention_dimension_wise(self, x):     # x.shape = (b,n,d) = (1024, 65, 384)
    #     # breakpoint()
    #     num_head = self.head
        
    #     x_chunked = list( x.chunk(num_head, dim=2))
        
    #     h0_qkv = self.h0_dim(x_chunked[0])      # x_chunked[0] : (b, n, d/h)
    #     h1_qkv = self.h1_dim(x_chunked[1])
    #     h2_qkv = self.h2_dim(x_chunked[2])
    #     h3_qkv = self.h3_dim(x_chunked[3])
    #     h4_qkv = self.h4_dim(x_chunked[4])
    #     h5_qkv = self.h5_dim(x_chunked[5])
    #     h6_qkv = self.h6_dim(x_chunked[6])
    #     h7_qkv = self.h7_dim(x_chunked[7])
    #     h8_qkv = self.h8_dim(x_chunked[8])
    #     h9_qkv = self.h9_dim(x_chunked[9])
    #     h10_qkv = self.h10_dim(x_chunked[10])
    #     h11_qkv = self.h11_dim(x_chunked[11])   # (b, n, d)
        
    #     h0_qkv_chunked = h0_qkv.chunk(3, dim=2)
    #     h1_qkv_chunked = h1_qkv.chunk(3, dim=2)
    #     h2_qkv_chunked = h2_qkv.chunk(3, dim=2)
    #     h3_qkv_chunked = h3_qkv.chunk(3, dim=2)
    #     h4_qkv_chunked = h4_qkv.chunk(3, dim=2)
    #     h5_qkv_chunked = h5_qkv.chunk(3, dim=2)
    #     h6_qkv_chunked = h6_qkv.chunk(3, dim=2)
    #     h7_qkv_chunked = h7_qkv.chunk(3, dim=2)
    #     h8_qkv_chunked = h8_qkv.chunk(3, dim=2)
    #     h9_qkv_chunked = h9_qkv.chunk(3, dim=2)
    #     h10_qkv_chunked = h10_qkv.chunk(3, dim=2)
    #     h11_qkv_chunked = h11_qkv.chunk(3, dim=2)   # (b, n, d/3)
        
        
    #     score0 = F.softmax(torch.einsum('bij, bkj -> bik', h0_qkv_chunked[0], h0_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn0 = torch.einsum('bij, bjk -> bik', score0, h0_qkv_chunked[2])   # (b,n,d/3)
        
    #     score1 = F.softmax(torch.einsum('bij, bkj -> bik', h1_qkv_chunked[0], h1_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn1 = torch.einsum('bij, bjk -> bik', score1, h1_qkv_chunked[2])   # (b,n,d/3)
        
    #     score2 = F.softmax(torch.einsum('bij, bkj -> bik', h2_qkv_chunked[0], h2_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn2 = torch.einsum('bij, bjk -> bik', score2, h2_qkv_chunked[2])   # (b,n,d/3)
        
    #     score3 = F.softmax(torch.einsum('bij, bkj -> bik', h3_qkv_chunked[0], h3_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn3 = torch.einsum('bij, bjk -> bik', score3, h3_qkv_chunked[2])   # (b,n,d/3)
        
    #     score4 = F.softmax(torch.einsum('bij, bkj -> bik', h4_qkv_chunked[0], h4_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn4 = torch.einsum('bij, bjk -> bik', score4, h4_qkv_chunked[2])   # (b,n,d/3)
        
    #     score5 = F.softmax(torch.einsum('bij, bkj -> bik', h5_qkv_chunked[0], h5_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn5 = torch.einsum('bij, bjk -> bik', score5, h5_qkv_chunked[2])   # (b,n,d/3)
        
    #     score6 = F.softmax(torch.einsum('bij, bkj -> bik', h6_qkv_chunked[0], h6_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn6 = torch.einsum('bij, bjk -> bik', score6, h6_qkv_chunked[2])   # (b,n,d/3)
        
    #     score7 = F.softmax(torch.einsum('bij, bkj -> bik', h7_qkv_chunked[0], h7_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn7 = torch.einsum('bij, bjk -> bik', score7, h7_qkv_chunked[2])   # (b,n,d/3)
        
    #     score8 = F.softmax(torch.einsum('bij, bkj -> bik', h8_qkv_chunked[0], h8_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn8 = torch.einsum('bij, bjk -> bik', score8, h8_qkv_chunked[2])   # (b,n,d/3)
        
    #     score9 = F.softmax(torch.einsum('bij, bkj -> bik', h9_qkv_chunked[0], h9_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn9 = torch.einsum('bij, bjk -> bik', score9, h9_qkv_chunked[2])   # (b,n,d/3)
        
    #     score10 = F.softmax(torch.einsum('bij, bkj -> bik', h10_qkv_chunked[0], h10_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn10 = torch.einsum('bij, bjk -> bik', score10, h10_qkv_chunked[2])   # (b,n,d/3)
        
    #     score11 = F.softmax(torch.einsum('bij, bkj -> bik', h11_qkv_chunked[0], h11_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
    #     attn11 = torch.einsum('bij, bjk -> bik', score11, h11_qkv_chunked[2])   # (b,n,d/3)
        
    #     all_attn = torch.cat([attn0, attn1, attn2, attn3, attn4, attn5,
    #                           attn6, attn7, attn8, attn9, attn10, attn11], dim=2)
    #     # breakpoint()
        
    #     out = self.out_dim(all_attn)
    #     out = self.dropout(out)
        
    #     return out
    
    def non_overlapping_head_attention_dimension_wise_3dk(self, x):
        num_head = self.head
        
        x_chunked = list( x.chunk(num_head, dim=2))
        
        h0_qkv = self.h0_dim_3dk(x_chunked[0])      # x_chunked[0] : (b, n, dk)
        h1_qkv = self.h1_dim_3dk(x_chunked[1])
        h2_qkv = self.h2_dim_3dk(x_chunked[2])
        h3_qkv = self.h3_dim_3dk(x_chunked[3])
        h4_qkv = self.h4_dim_3dk(x_chunked[4])
        h5_qkv = self.h5_dim_3dk(x_chunked[5])
        h6_qkv = self.h6_dim_3dk(x_chunked[6])
        h7_qkv = self.h7_dim_3dk(x_chunked[7])
        h8_qkv = self.h8_dim_3dk(x_chunked[8])
        h9_qkv = self.h9_dim_3dk(x_chunked[9])
        h10_qkv = self.h10_dim_3dk(x_chunked[10])
        h11_qkv = self.h11_dim_3dk(x_chunked[11])   # (b, n, 3dk)

        
        h0_qkv_chunked = h0_qkv.chunk(3, dim=2)
        h1_qkv_chunked = h1_qkv.chunk(3, dim=2)
        h2_qkv_chunked = h2_qkv.chunk(3, dim=2)
        h3_qkv_chunked = h3_qkv.chunk(3, dim=2)
        h4_qkv_chunked = h4_qkv.chunk(3, dim=2)
        h5_qkv_chunked = h5_qkv.chunk(3, dim=2)
        h6_qkv_chunked = h6_qkv.chunk(3, dim=2)
        h7_qkv_chunked = h7_qkv.chunk(3, dim=2)
        h8_qkv_chunked = h8_qkv.chunk(3, dim=2)
        h9_qkv_chunked = h9_qkv.chunk(3, dim=2)
        h10_qkv_chunked = h10_qkv.chunk(3, dim=2)
        h11_qkv_chunked = h11_qkv.chunk(3, dim=2)   # (b, n, dk)
        
        
        score0 = F.softmax(torch.einsum('bij, bkj -> bik', h0_qkv_chunked[0], h0_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn0 = torch.einsum('bij, bjk -> bik', score0, h0_qkv_chunked[2])   # (b,n,dk)
        
        score1 = F.softmax(torch.einsum('bij, bkj -> bik', h1_qkv_chunked[0], h1_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn1 = torch.einsum('bij, bjk -> bik', score1, h1_qkv_chunked[2])   # (b,n,dk)
        
        score2 = F.softmax(torch.einsum('bij, bkj -> bik', h2_qkv_chunked[0], h2_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn2 = torch.einsum('bij, bjk -> bik', score2, h2_qkv_chunked[2])   # (b,n,dk)
        
        score3 = F.softmax(torch.einsum('bij, bkj -> bik', h3_qkv_chunked[0], h3_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn3 = torch.einsum('bij, bjk -> bik', score3, h3_qkv_chunked[2])   # (b,n,dk)
        
        score4 = F.softmax(torch.einsum('bij, bkj -> bik', h4_qkv_chunked[0], h4_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn4 = torch.einsum('bij, bjk -> bik', score4, h4_qkv_chunked[2])   # (b,n,dk)
        
        score5 = F.softmax(torch.einsum('bij, bkj -> bik', h5_qkv_chunked[0], h5_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn5 = torch.einsum('bij, bjk -> bik', score5, h5_qkv_chunked[2])   # (b,n,dk)
        
        score6 = F.softmax(torch.einsum('bij, bkj -> bik', h6_qkv_chunked[0], h6_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn6 = torch.einsum('bij, bjk -> bik', score6, h6_qkv_chunked[2])   # (b,n,dk)
        
        score7 = F.softmax(torch.einsum('bij, bkj -> bik', h7_qkv_chunked[0], h7_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn7 = torch.einsum('bij, bjk -> bik', score7, h7_qkv_chunked[2])   # (b,n,dk)
        
        score8 = F.softmax(torch.einsum('bij, bkj -> bik', h8_qkv_chunked[0], h8_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn8 = torch.einsum('bij, bjk -> bik', score8, h8_qkv_chunked[2])   # (b,n,dk)
        
        score9 = F.softmax(torch.einsum('bij, bkj -> bik', h9_qkv_chunked[0], h9_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn9 = torch.einsum('bij, bjk -> bik', score9, h9_qkv_chunked[2])   # (b,n,dk)
        
        score10 = F.softmax(torch.einsum('bij, bkj -> bik', h10_qkv_chunked[0], h10_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn10 = torch.einsum('bij, bjk -> bik', score10, h10_qkv_chunked[2])   # (b,n,dk)
        
        score11 = F.softmax(torch.einsum('bij, bkj -> bik', h11_qkv_chunked[0], h11_qkv_chunked[1])/self.sqrt_d, dim=-1)   # (b,n,n)
        attn11 = torch.einsum('bij, bjk -> bik', score11, h11_qkv_chunked[2])   # (b,n,dk)
        
        all_attn = torch.cat([attn0, attn1, attn2, attn3, attn4, attn5,
                              attn6, attn7, attn8, attn9, attn10, attn11], dim=2)
        
        
        out = self.out_dim_3dk(all_attn)
        out = self.dropout(out)
        
        return out
        
        
        
        

        
        
        

        
    def generate_idx_for_head(self, num_token, type_idx='sequential_order' ):
        
        if type_idx == 'sequential_order':
            idx = torch.arange(num_token)
            
        elif type_idx == 'random_order':
            idx = torch.randperm(num_token)
            
        return idx 
        

        
    
    
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feats:int, head:int=8, dropout:float=0.):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.head = head
#         self.feats = feats
#         self.sqrt_d = self.feats**0.5

#         self.q = nn.Linear(feats, feats)
#         self.k = nn.Linear(feats, feats)
#         self.v = nn.Linear(feats, feats)

#         self.o = nn.Linear(feats, feats)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         b, n, f = x.size()
#         q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)       # self.q(x) 의 output에 .view
#         k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
#         v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)       # 최종: 현재 q,k,v shape: (batch_size, head, seq_len, head_dim)
#                                                                                         #                        (b, h, n, head_dim)
#         # attention 계산 part1. softmax(QK/d)
        
#         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1)   #(b,h,n,n)
#         # attention 계산 part2. softmax(QK/d)*V 
#         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        
#         # output projection layer 계산 O
#         tmp = attn.flatten(2)
#         tmp = self.o(tmp)
#         o = self.dropout(tmp)
#         # o = self.dropout(self.o(attn.flatten(2)))
        
#         return o

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



