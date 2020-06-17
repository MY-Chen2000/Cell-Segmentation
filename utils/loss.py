import torch.nn.functional as F

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    # print(target)
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    #input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    input = input.permute(0,2,3,1).contiguous().view(-1, c)
    #print('input:\t{}'.format(input.shape))
    #print('target:\t{}'.format(target.shape))
    #input=input.view(-1)
    target = target.view(-1)
    #print('target2:\t{}'.format(target.shape))
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss