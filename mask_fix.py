"""ComfyUI node: fix SAM3 video masks packed with MiniMax H3 AV latents.
Place in ComfyUI/custom_nodes/avatar_h3_mask_fix.py and restart ComfyUI.
"""
import torch
import torch.nn.functional as F
import comfy.nested_tensor

class AvatarH3FixMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "fix"
    CATEGORY = "MiniMax H3/SAM3"

    def fix(self, latent):
        result = latent.copy()
        samples = result["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("Avatar H3 Fix Mask: expected a MiniMax H3 nested VIDEO+AUDIO latent")
        streams = list(samples.unbind())
        if len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
            raise ValueError("Avatar H3 Fix Mask: expected video [B,C,T,H,W] and audio [B,C,S,T]")
        raw = result.get("noise_mask")
        if raw is None or not getattr(raw, "is_nested", False):
            raise ValueError("Avatar H3 Fix Mask: no video+audio mask found; check upstream mask and concat nodes")
        masks = list(raw.unbind())
        if len(masks) != 2:
            raise ValueError("Avatar H3 Fix Mask: expected exactly two masks")
        m, a = masks
        b, _, t, h, w = streams[0].shape
        # SetLatentNoiseMask reshapes image/video MASK [T,H,W] into [T,1,H,W].
        # SelfLift Avatar interprets 4D as [B,C,H,W] and therefore rejects T as B.
        if m.ndim == 4 and m.shape[0] == t and m.shape[1] == 1 and b == 1:
            m = m.permute(1,0,2,3).unsqueeze(0)  # [1,1,T,H,W]
        elif m.ndim == 3 and m.shape[0] == t and b == 1:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 5 and m.shape[0] in (1,b) and m.shape[2] in (1,t):
            pass
        elif m.ndim == 4 and m.shape[0] in (1,b):
            m = m.unsqueeze(2)  # static mask over time
        else:
            raise ValueError(f"Avatar H3 Fix Mask: video mask {tuple(m.shape)} cannot be aligned to {tuple(streams[0].shape)}")
        if m.shape[-2:] != (h,w):
            x = m.reshape(-1,1,*m.shape[-2:]).float()
            m = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=False).reshape(*m.shape[:-2],h,w)
        m = m.float().clamp(0,1)
        if a.ndim == 4 and a.shape[0] in (1,streams[1].shape[0]):
            # Keep the existing constant source-audio mask; SelfLift Avatar handles broadcasting.
            a = a.float().clamp(0,1)
        else:
            raise ValueError(f"Avatar H3 Fix Mask: unexpected audio mask shape {tuple(a.shape)}")
        result["noise_mask"] = comfy.nested_tensor.NestedTensor([m,a])
        print(f"[AvatarH3FixMask] video mask {tuple(m.shape)}, audio mask {tuple(a.shape)}")
        return (result,)

NODE_CLASS_MAPPINGS = {"AvatarH3FixMask": AvatarH3FixMask}
NODE_DISPLAY_NAME_MAPPINGS = {"AvatarH3FixMask": "Avatar H3 · Fix SAM3 Mask BCTHW"}
