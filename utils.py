import torch

def save_checkpoint(checkpoint_path, policy_model, optimizer, sketch_encoder_model=None, optimizer2 = None):
        torch.save({'optimizer1_state_dict': optimizer.state_dict(),
                    'policy_old_state_dict': policy_model.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'sketch_encoder_state_dict': sketch_encoder_model.state_dict()}, checkpoint_path)
        
def load_checkpoint(checkpoint_path, policy_model, optimizer, old_policy_model, sketch_encoder_model=None, optimizer2 = None, device=None):
        ### load checkpoint
        if device is not None:
            _ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
        else:
            _ckpt = torch.load(checkpoint_path)

        ### load state dicts
        policy_model.load_state_dict(_ckpt['policy_old_state_dict'])
        old_policy_model.load_state_dict(_ckpt['policy_old_state_dict'])
        optimizer.load_state_dict(_ckpt['optimizer1_state_dict'])
        if sketch_encoder_model is not None:
            sketch_encoder_model.load_state_dict(_ckpt['sketch_encoder_state_dict'])
        if optimizer2 is not None:
            optimizer2.load_state_dict(_ckpt['optimizer2_state_dict'])
        
        #self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        #self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        return policy_model, optimizer, old_policy_model, sketch_encoder_model, optimizer2