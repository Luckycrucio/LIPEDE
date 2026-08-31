# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


def _checkpoint_state(checkpoint):
    """Return model weights from all checkpoint formats used by this project."""
    if isinstance(checkpoint, dict):
        for key in ('checkpoint', 'model_state_dict', 'state_dict'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _compatible_state_dict(model, checkpoint_state):
    """Match legacy SpConv/DDP/compiled-RWKV keys against ``model``.

    The TorchSparse adapter deliberately retains SpConv convolution parameter
    names and shapes.  The only key differences we need to tolerate are
    wrappers introduced by DDP (``module.``) and torch.compile
    (``_orig_mod.``).  Shape checking prevents an accidental partial load.
    """
    target = model.state_dict()
    matched = {}
    skipped = []
    for original_key, value in checkpoint_state.items():
        candidates = [original_key]
        key = original_key
        if key.startswith('module.'):
            candidates.append(key[7:])
        candidates.extend(candidate.replace('._orig_mod.', '.') for candidate in candidates[:])
        for candidate in candidates:
            if candidate in target and target[candidate].shape == value.shape:
                matched[candidate] = value
                break
        else:
            skipped.append(original_key)
    return matched, skipped


def load_checkpoint_compatible(model_load_path, model, device=None, return_mask=False):
    """Load a SpConv-era or TorchSparse-era checkpoint without losing weights."""
    checkpoint = torch.load(model_load_path, map_location=device)
    matched, skipped = _compatible_state_dict(model, _checkpoint_state(checkpoint))
    status = model.load_state_dict(matched, strict=True)
    print("STATUS", status)
    print('matched parameter sets: {}, skipped: {}'.format(len(matched), len(skipped)))
    if skipped:
        print('first skipped keys:', skipped[:10])
    if return_mask:
        return model, checkpoint.get('mask') if isinstance(checkpoint, dict) else None
    return model


def load_checkpoint(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            match_size += 1
            part_load[k] = value
        else:
            print(k in my_model_dict)
            print(my_model_dict[k].shape, value.shape)
            assert len(value.shape) == 1 or len(value.shape) == 5
            if len(value.shape) == 1:
                c = value.shape[0]
                cc = my_model_dict[k].shape[0] - c #int(c*0.5)
                if cc <= c:
                    value = torch.cat([value, value[:cc]], dim=0)
                else:
                    value = torch.cat([value, value, value[:(cc-c)]], dim=0)
            else:
                _, _, _, c1, c2 = value.shape
                cc1 = my_model_dict[k].shape[3] - c1 #int(c1*0.5)
                cc2 = my_model_dict[k].shape[4] - c2 #int(c2*0.5)
                if cc1 > 0 and cc1 <= c1:
                    value1 = torch.cat([value, value[:, :, :, :cc1, :]], dim=3) 
                elif cc1 > c1:
                    value1 = torch.cat([value, value, value[:, :, :, :(cc1-c1), :]], dim=3) 
                else:
                    value1 = value
                if cc2 > 0 and cc2 <= c2:
                    value = torch.cat([value1, value1[:, :, :, :, :cc2]], dim=4) 
                elif cc2 > c2:
                    value = torch.cat([value1, value1, value1[:, :, :, :, :(cc2-c2)]], dim=4) 
                else:
                    value = value1
            nomatch_size += 1
            part_load[k] = value
            assert my_model_dict[k].shape == value.shape
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model

def load_checkpoint_old(model_load_path, model):
    return load_checkpoint_compatible(model_load_path, model)

def load_checkpoint_model_mask(model_load_path, model, device):
    return load_checkpoint_compatible(model_load_path, model, device, return_mask=True)

def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
