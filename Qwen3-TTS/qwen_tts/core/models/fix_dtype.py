import torch
import torch.nn as nn

# Store original __init__ method
original_SqueezeExcitationBlock_init = None

def patched_SqueezeExcitationBlock_init(self, in_channels, se_channels, out_channels):
    """Patched __init__ to fix dtype issues in SqueezeExcitationBlock."""
    # Call original __init__
    original_SqueezeExcitationBlock_init(self, in_channels, se_channels, out_channels)
    
    # Fix bias dtype to match expected dtype (bfloat16)
    # The model uses mixed precision bf16, so bias should be in bf16
    if hasattr(self, 'bias') and self.bias is not None:
        # Get the target dtype from the model if available
        target_dtype = torch.bfloat16
        if hasattr(self, 'conv'):
            # Try to get dtype from conv layer
            target_dtype = self.conv.weight.dtype
        else:
            target_dtype = torch.bfloat16
        
        # Cast bias to correct dtype
        self.bias.data = self.bias.data.to(target_dtype)
    
    # Also fix conv layers dtype if they're float32
    if hasattr(self, 'conv1'):
        self.conv1.data = self.conv1.data.to(target_dtype)
    if hasattr(self, 'conv2'):
        self.conv2.data = self.conv2.data.to(target_dtype)

# Apply monkey patch
def apply_patch():
    """Apply the monkey patch to fix dtype issues."""
    from qwen_tts.core.models.modeling_qwen3_tts import SqueezeExcitationBlock
    
    # Patch SqueezeExcitationBlock
    SqueezeExcitationBlock.__init__ = patched_SqueezeExcitationBlock_init
    
    print("✓ Dtype fix for SqueezeExcitationBlock applied successfully")

if __name__ == "__main__":
    apply_patch()
