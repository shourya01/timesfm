from typing import OrderedDict

class EmptyDict(Exception):
    pass

def save_learnable_state_dict(model, name_filter, *args, **kwargs):

    if isinstance(name_filter, str):
        name_filter = [name_filter]
    full_state_dict = model.state_dict(*args, **kwargs)
    return_dict = { k: v for k, v in full_state_dict.items() if any(substr in k for substr in name_filter)}

    if not return_dict:
        raise EmptyDict("Trying to save an empty dict!")

    return return_dict