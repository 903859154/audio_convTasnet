import torchaudio

def get_model(
    num_sources,
    enc_kernel_size=16,# L
    enc_num_feats=512, # N

    msk_kernel_size=3, # P
    msk_num_feats=128, # B
    msk_num_hidden_feats=512, # H

    #network config
    msk_num_layers=8, # X 内部层网络
    msk_num_stacks=3, # R 外部层网络  总网络等于 外 × 内 = 4 × 8
    msk_activate="relu",
):
    model = torchaudio.models.ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
        msk_activate=msk_activate,
    )
    return model
