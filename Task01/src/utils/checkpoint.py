import os
import shutil
import torch

def save_checkpoint(state, is_best, filename_prefix="checkpoint", artefact_domain=""):
    artefact_filename_part = f"_{artefact_domain}" if artefact_domain else ""
    filename = f"{filename_prefix}{artefact_filename_part}.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{filename_prefix}{artefact_filename_part}_best.pth.tar")
        print(f"Saved new best model for {artefact_domain} to {filename}_best.pth.tar")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', 0)
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_metric {best_metric:.4f})")
        return start_epoch, best_metric
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return 0, 0.0 