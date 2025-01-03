import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3, stride=1, padding=1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                # nn.LayerNorm(int(outchannel / 1),dim=1)
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return F.relu(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads = 8, dropout = 0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias = False)       
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2, x3):
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_k(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)#(b,n,dim)

class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout = 0., norm_first = False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads = heads, dropout = dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout = dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + x1)
            x = self.norm2(self.feedforward(x) + x)

        return x

# HSA Attention Module + Res Block
class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    (3, [resolution, patch_size, mm_hidden_size], 8, mm_hidden_size, 512, 0.1)
    """
    def __init__(self, n_layers, feature_size, heads, hidden_dim, attention_dim = 512, dropout = 0.):
        super(AttentiveEncoder, self).__init__()
        image_size_sig, patch_size, channels = feature_size
        # print(feature_size)
        if image_size_sig == -1:
            image_size_sig = 384
        self.h_feat = image_size_sig // patch_size
        # print('self.h_feat:', self.h_feat)
        self.w_feat = self.h_feat
        self.h_embedding = nn.Embedding(self.h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(self.w_feat, int(channels/2))
        self.selftrans = nn.ModuleList([])
        for i in range(n_layers):                 
            self.selftrans.append(nn.ModuleList([
                Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
                Transformer(channels*2, channels*2, heads, attention_dim, hidden_dim, dropout, norm_first=False),
            ]))
        self.Conv1 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.LN = resblock(hidden_dim, hidden_dim)
        self.cos = torch.nn.CosineSimilarity(dim=1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img1, img2):
        batch, Len_feat, dep = img1.shape
        pos_h = torch.arange(self.h_feat).cuda()
        pos_w = torch.arange(self.w_feat).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(self.h_feat, 1, 1),
                                       embed_h.unsqueeze(1).repeat(1, self.w_feat, 1)],
                                       dim = -1)                            
        pos_embedding = pos_embedding.permute(2,0,1).unsqueeze(0).repeat(batch, 1, 1, 1)# (batch, d_model, h, w)
        pos_embedding = pos_embedding.view(batch, dep, -1).transpose(-1, 1)
        img1 = img1 + pos_embedding
        img2 = img2 + pos_embedding
        img_sa1, img_sa2 = img1, img2

        for (l, m) in self.selftrans:           
            img_sa1 = l(img_sa1, img_sa1, img_sa1) + img_sa1
            img_sa2 = l(img_sa2, img_sa2, img_sa2) + img_sa2
            img = torch.cat([img_sa1, img_sa2], dim = -1)
            img = m(img, img, img)
            img_sa1 = img[:,:,:dep] + img1
            img_sa2 = img[:,:,dep:] + img2

        # img1 = img_sa1.reshape(batch, self.h_feat, self.w_feat, dep).transpose(-1, 1)
        # img2 = img_sa2.reshape(batch, self.h_feat, self.w_feat, dep).transpose(-1, 1)

        img1 = img_sa1.reshape(batch, self.h_feat, self.w_feat, dep).permute(0, 3, 1, 2)
        img2 = img_sa2.reshape(batch, self.h_feat, self.w_feat, dep).permute(0, 3, 1, 2)

        img_sam = self.cos(img1, img2)
        # print('img_sam shape', img_sam.shape)
        x = torch.cat([img1, img2], dim=1) + img_sam.unsqueeze(1)
        # print('concatenated x shape', x.shape)
        x = self.LN(self.Conv1(x))
        # print('x after ResBlock shape: ',x.shape)
        batch, channel = x.size(0), x.size(1)
        img = x.view(batch, channel, -1).permute(0, 2, 1) # batch, length, channel

        return img