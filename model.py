''''
Implementation of Ppoint Transformer-in-Transformer, insipered by the
Transformer-in-Transformer implementation by lucidrains
https://github.com/lucidrains/transformer-in-transformer

Author: Axel Berg
'''

import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from helpers import farthest_point_sample, index_points, get_graph_feature


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=3,
        dim_head=64,
        dropout=0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class PointTNT(nn.Module):
    '''
    Point Transformer-in-Transformer that applies self-attention hierarchically
    on edges around anchor points, and on the anchor points themselves
    '''

    def __init__(
        self,
        args,
        num_classes,
        channels=3,
    ):
        super(PointTNT, self).__init__()

        self.local_attention = args.local_attention
        self.global_attention = args.global_attention
        self.n_anchor = args.n_anchor
        self.n_neigh = args.k
        self.dilation = args.dilation

        # Projection to point dimension
        self.to_point_tokens = nn.Sequential(
            Rearrange('b c (p) (n) -> (b p) n c'),
            nn.Linear(channels, args.point_dim)
        )

        # Projection to patch dimension
        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, args.patch_dim)
        )

        layers = nn.ModuleList([])
        for _ in range(args.depth):

            point_to_patch = nn.Sequential(
                nn.LayerNorm(2 * args.point_dim),
                nn.Linear(2 * args.point_dim, args.patch_dim),
            )

            layers.append(nn.ModuleList([
                # Local Transformer module
                PreNorm(args.point_dim, Attention(dim=args.point_dim, heads=args.heads, dim_head=args.dim_head,
                        dropout=args.attn_dropout)) if self.local_attention else nn.Identity(),
                PreNorm(args.point_dim, FeedForward(
                    dim=args.point_dim, dropout=args.ff_dropout)),
                # Projection from point_dim to patch_dim
                point_to_patch,
                # Global Transformer module
                PreNorm(args.patch_dim, Attention(dim=args.patch_dim, heads=args.heads, dim_head=args.dim_head,
                        dropout=args.attn_dropout)) if self.global_attention else nn.Identity(),
                PreNorm(args.patch_dim, FeedForward(
                    dim=args.patch_dim, dropout=args.ff_dropout)),
            ]))

        self.layers = layers

        # Combine intermediate features and project to to emb_dim
        self.final_conv = nn.Sequential(
            nn.LayerNorm(args.patch_dim * args.depth),
            nn.Linear(args.patch_dim * args.depth, args.emb_dims),
            nn.GELU(),
        )

        # Classifier head
        self.mlp_head = nn.Sequential(
            nn.Linear(args.emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        batch_size, _, _ = x.shape

        # Sample anchor points
        fps_idx = farthest_point_sample(
            xyz=x.permute(0, 2, 1), npoint=self.n_anchor)
        anchors = index_points(x.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
        torch.cuda.empty_cache()

        # Project to patch dimension
        patches = self.to_anchor(anchors)

        # Get edge features (x_i - x_j) for all points
        e = get_graph_feature(x, k=self.n_neigh, d=self.dilation)
        e = e[:, 0:3, :, :]

        # Take the edge features only for anchors
        e = index_points(e.permute(0, 2, 1, 3), fps_idx).permute(0, 2, 1, 3)

        # Project edge features to point_dim
        edges = self.to_point_tokens(e)
        b, _, _ = edges.shape

        # list to store intermediate features from each transformer layer
        ylist = []

        for point_attn, point_ff, point_to_patch_residual, patch_attn, patch_ff in self.layers:

            # apply local transformer to edge features within patches
            edges = point_attn(edges) + edges
            edges = point_ff(edges) + edges

            # pool over edges
            p1 = edges.max(dim=1, keepdim=False)[0]
            p2 = edges.mean(dim=1, keepdim=False)
            edge_features = torch.cat((p1, p2), dim=1)

            # project to patch_dimension and add edge features to patch features
            patches_residual = point_to_patch_residual(edge_features)
            patches_residual = rearrange(
                patches_residual, '(b p) d -> b p d', b=batch_size)
            patches = patches + patches_residual

            # apply global transformer
            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches

            # save intermediate patch features
            ylist.append(patches)

        # apply final layer to all intermediate patch features
        y = torch.cat(ylist, dim=-1)
        y = self.final_conv(y)

        # pool over all patch features
        y1 = y.max(dim=1, keepdim=False)[0]
        y2 = y.mean(dim=1, keepdim=False)
        y = torch.cat((y1, y2), dim=1)

        # apply classifier head
        return self.mlp_head(y)


class Baseline(nn.Module):
    '''
    Baseline Transformer model that applies self-attention globally on all points
    '''

    def __init__(
        self,
        args,
        num_classes,
        channels=3,
    ):
        super(Baseline, self).__init__()

        self.global_attention = args.global_attention

        self.to_anchor = nn.Sequential(
            Rearrange('b c n -> b n c'),
            nn.Linear(channels, args.patch_dim)
        )

        layers = nn.ModuleList([])
        for _ in range(args.depth):

            layers.append(nn.ModuleList([
                PreNorm(args.patch_dim, Attention(dim=args.patch_dim, heads=args.heads, dim_head=args.dim_head,
                        dropout=args.attn_dropout)) if self.global_attention else nn.Identity(),
                PreNorm(args.patch_dim, FeedForward(
                    dim=args.patch_dim, dropout=args.ff_dropout)),
            ]))

        self.layers = layers

        self.final_conv = nn.Sequential(
            nn.LayerNorm(args.patch_dim * args.depth),
            nn.Linear(args.patch_dim * args.depth, args.emb_dims),
            nn.GELU(),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(args.emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        batch_size, _, _ = x.shape

        # all points are anchores
        points = self.to_anchor(x)
        ylist = []

        for patch_attn, patch_ff in self.layers:

            # apply global transformer
            points = patch_attn(points) + points
            points = patch_ff(points) + points

            # store intermediate features
            ylist.append(points)

        # apply final layer to all intermediate patch features
        y = torch.cat(ylist, dim=-1)
        y = self.final_conv(y)

        # pool over all points
        y1 = y.max(dim=1, keepdim=False)[0]
        y2 = y.mean(dim=1, keepdim=False)
        y = torch.cat((y1, y2), dim=1)

        # apply classifier head
        return self.mlp_head(y)
