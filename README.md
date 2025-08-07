Input (이미지 배치) → model.forward()
    ⤷ conv1 → bn1 → relu → maxpool
    ⤷ layer1: 2개 BasicBlock
    ⤷ layer2: 2개 BasicBlock (downsample O)
    ⤷ layer3: 2개 BasicBlock (downsample O)
    ⤷ layer4: 2개 BasicBlock (downsample O)
    ⤷ avgpool → flatten → fc (logits)