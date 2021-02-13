import pickle
import torch
import torch.distributed


def all_gather_list(data, max_size=4096):
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size) for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255
    in_buffer[1] = enc_size % 255
    in_buffer[2: enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffers = out_buffers[i]
        size = (255 * out_buffers[0].item()) + out_buffers[1].item()
        bytes_list = bytes(out_buffers[2: size+2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)

    return results
