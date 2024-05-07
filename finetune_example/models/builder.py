import os
import sys
import mlflow
import torch
from .core import PepCrosser


def model_factory(cfg):

    print("load pepcrosser model")
    model = PepCrosser(cfg)

    for name, param in model.named_parameters():
        print(name, param.shape)
        if param.requires_grad:
            # check if the param is nan
            if torch.isnan(param.data).any():
                print("nan found in param: {}".format(name))
                sys.exit(1)

    if cfg.model.pretrained:
        model_dict = model.state_dict()
        model_path = os.path.join(cfg.orig_cwd, cfg.model.pretrained)
        print("loading weights from : {}".format(model_path))
        pretrained_model = mlflow.pytorch.load_model(model_path,
                                                     map_location="cpu")
        # for param in model.parameters():
        #     print(param.requires_grad)

        best_state_dict = {
            k.replace("module.", ""): v
            for (k, v) in list(pretrained_model.state_dict().items())
        }

        for k, v in best_state_dict.items():
            if k in model_dict:
                print("updating parameters for: {}".format(k))
                model_dict[k] = v

        model.load_state_dict(model_dict)

    # check layers name and grad status
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model
