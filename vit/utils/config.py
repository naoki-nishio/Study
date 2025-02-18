class Config:
    # モデルパラメータ
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 10         # CIFAR-10 用に 10 クラス
    embed_dim = 256          # 埋め込み次元（用途に合わせて調整）
    depth = 6                # Transformer レイヤの数
    num_heads = 8            # Multi-head Attention のヘッド数
    mlp_ratio = 4.0
    dropout_rate = 0.1

    # 学習パラメータ
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    weight_decay = 0.0001
    data_root = './data'
