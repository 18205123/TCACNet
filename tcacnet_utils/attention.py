import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyLoss(nn.Module):

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def left_coordinates(indexes, feature_w, device):
    
    indexes = indexes.long().to(device)
    idx_x = indexes // feature_w
    idx_y = torch.fmod(indexes, feature_w)
    x_left = idx_x.view(-1, 1)
    y_left = (90*idx_y).view(-1, 1)
    coords = torch.cat((x_left, y_left), 1)
    
    return coords, idx_x, idx_y

def salient_indices(grads, batch_size, n_slices):
    
    M = torch.sqrt((torch.sum(-grads, dim=1)).pow(2))
    M_reshape = M.view(batch_size, -1)
    values, indices = torch.topk(M_reshape, n_slices, largest=False)

    assert indices.shape == (batch_size, n_slices)

    return indices

def extract_slices(inputs, batch_size, left_coord, device):
    
    slices = []
    for i in range(batch_size):
        sliced = inputs[i].narrow(1, left_coord[i, 0], 44)
        sliced = sliced.narrow(2, left_coord[i, 1], 199)
        slices.append(sliced)
    slices = torch.stack(slices).to(device)

    return slices

def extract_one_feature(inputs, indexes, feature_w, batch_size, device):
       
    left_coord, idx_i, idx_j = left_coordinates(indexes, feature_w, device)
    slices = extract_slices(inputs, batch_size, left_coord, device)
    src_idxs = torch.cat((idx_i.view(-1, 1), idx_j.view(-1, 1)), 1).to(device)

    return src_idxs, slices, left_coord


def extract_features(inputs, model_local, batch_size, k_indexes, feature_w, n_slices, device):

    k_src_idxs = []
    k_slices = []
    for i in range(n_slices):
        src_idxs, slices, left_coord = extract_one_feature(inputs, k_indexes[:, i], feature_w, batch_size, device)
        k_src_idxs.append(src_idxs)
        k_slices.append(slices)
    concat_slices = torch.cat(k_slices, 0).to(device)
    assert concat_slices.shape == (batch_size * n_slices, 1, 44, 199)
    concat_k_features = model_local(concat_slices)
    k_features = torch.split(concat_k_features, n_slices, 0)

    return k_features, k_src_idxs

def replace_features(global_features, local_features, replace_idxs, device):

    batch_size, feature_ch, feature_h, feature_w = global_features.size()

    def _convert_to_1d_idxs(src_idxs):
        batch_idx_len = feature_ch * feature_w * feature_h
        batch_idx_base = (torch.Tensor([i * batch_idx_len for i in range(batch_size)]).long()).to(device)
        batch_1d = feature_ch * feature_w * src_idxs[:, 0] + feature_ch * src_idxs[:, 1]
        batch_1d = torch.add(batch_1d, batch_idx_base)
        flat_idxs = [batch_1d + i for i in range(feature_ch)]
        flat_idxs = (torch.stack(flat_idxs)).t()
        flat_idxs = flat_idxs.contiguous().view(-1)
        return flat_idxs

    flat_global_features = global_features.view(-1)
    flat_local_features = [i.view(-1) for i in local_features]
    flat_local_features = torch.cat(flat_local_features, 0)
    flat_local_idxs = [_convert_to_1d_idxs(i) for i in replace_idxs]
    flat_local_idxs = torch.cat(flat_local_idxs, 0)
    flat_global_replaced = torch.gather(flat_global_features, 0, flat_local_idxs)
    if flat_global_replaced.size() != flat_local_features.size():
        print('Assertion error : flat_global_replaced.size()', flat_global_replaced.size(),
              ' !=  flat_local_features.size()', flat_local_features.size())
        assert flat_global_replaced.size() == flat_local_features.size()
    merged = flat_global_features
    merged.clone()[flat_local_idxs] = flat_local_features
    merged = merged.view(global_features.size())

    return merged, flat_global_replaced, flat_local_features

def inference(inputs, wpser, model_global, model_local, model_top, n_slices, device, only_global_model, is_training):
    
    if only_global_model:
        global_features = model_global(inputs)
        output_global = model_top(global_features)
        return output_global, torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)
    
    d_ent_d_features = torch.Tensor().to(device)
    with torch.enable_grad():
        conv2_weight = model_global.conv2.weight.reshape(-1, 44)
        conv2_weight_abs = torch.abs(conv2_weight)
        channel_w = conv2_weight_abs.mean(dim = 0)
        channel_per = wpser
        channel_per_m = channel_per.mean(dim = 0)
        channel_per_m = channel_per_m[0,:]
        two_22 = torch.full((44,),2.0)
        two_ex = torch.pow(two_22.to(device), 1/channel_per_m)
        channel_loss = 1*torch.sum(two_ex*channel_w, dim = 0)
        
        entropy_loss_fn = EntropyLoss()
        global_features = model_global(inputs)
        output_global = model_top(global_features)
        entropy = entropy_loss_fn(output_global)
        
        d_ent_d_features = torch.autograd.grad(entropy, global_features, grad_outputs=torch.ones(entropy.size()).to(device))
    
    d_entropy_d_features = torch.cat(d_ent_d_features, 0)
    batch_size, feature_ch, feature_h, feature_w = global_features.size()
    top_k_indices = salient_indices(d_entropy_d_features, batch_size, n_slices)
    local_features, local_indexes = extract_features(inputs, model_local, batch_size, top_k_indices, feature_w, n_slices, device)
    merged, flat_global_replaced, flat_local_features = replace_features(global_features, local_features, local_indexes, device)
    raw_hint_loss = torch.sum((flat_global_replaced - flat_local_features).pow(2), dim=0)
    hint_loss = 0.05 * raw_hint_loss / (batch_size * n_slices)
    output_merged = model_top(merged)
    
    return output_merged, hint_loss, channel_loss