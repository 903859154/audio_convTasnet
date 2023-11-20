import torchaudio

def get_model(
    num_sources,
    enc_kernel_size=16,
    enc_num_feats=512, # N

    msk_kernel_size=3,
    msk_num_feats=128,
    msk_num_hidden_feats=512,

    #network config
    msk_num_layers=8,
    msk_num_stacks=3,
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