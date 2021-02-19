def tile(x, count, dim=0):
    """
    将 x 在 dim 维铺开 count 次
    tile([[1, 2], [3, 4]], 2, dim=0) => [[1, 2], [1, 2], [3, 4], [3, 4]]
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm)
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    # 先对 x 做 transpose 可以使 repeat 在一起
    # 否则 [[1, 2], [3, 4], [1, 2], [3, 4]]
    x = x.contiguous() \
         .view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
