import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MPB(nn.Module):
    """Modified MPB to predict 512×512 masks from 256×256 features"""
    def __init__(self, in_channels=64):  # 64 for features[1]
        super(MPB, self).__init__()
        
        # Encoder-decoder with MORE upsampling for 512×512 output
        self.conv1 = ConvBlock(in_channels, 128, 3, 2, 1)      # 256→128
        self.conv2 = ConvBlock(128, 256, 3, 2, 1)              # 128→64
        
        # MORE decoder layers to reach 512×512
        self.upconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)    # 64→128
        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)     # 128→256  
        self.upconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)      # 256→512  ← NEW LAYER
        
        # Final mask prediction
        self.mask_conv = nn.Conv2d(64, 1, 3, 1, 1)              # CHANGED from 128 to 64
        self.sigmoid = nn.Sigmoid()
        
        # Batch normalization for ALL upsampling layers
        self.bn_up1 = nn.BatchNorm2d(128)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.bn_up3 = nn.BatchNorm2d(64)  
    
    def forward(self, x):  # x should be [B, 64, 256, 256]
        # Encoder
        x1 = self.conv1(x)          # [B, 128, 128, 128]
        x2 = self.conv2(x1)         # [B, 256, 64, 64]
        
        # Decoder with THREE upsampling steps
        x = F.relu(self.bn_up1(self.upconv1(x2)))  # [B, 128, 128, 128]
        x = F.relu(self.bn_up2(self.upconv2(x)))   # [B, 64, 256, 256]
        x = F.relu(self.bn_up3(self.upconv3(x)))   # [B, 32, 512, 512]  ← NEW
        
        # Generate mask
        mask = self.sigmoid(self.mask_conv(x))     # [B, 1, 512, 512]  ← TARGET SIZE
        return mask
    
    


class GatedConv2d(nn.Module):
    """Gated Convolution for Local Attentive Module"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super(GatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feature = self.conv(x)
        mask = self.sigmoid(self.mask_conv(x))
        return feature * mask


class LocalAttentiveModule(nn.Module):
    """Local Attentive Module (LAM) for capturing local semantics and textures"""
    def __init__(self, in_channels: int, out_channels: int):
        super(LocalAttentiveModule, self).__init__()
        self.gconv = GatedConv2d(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.gconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MutualEncoder(nn.Module):
    """Mutual Encoder shared by MPB and LRB"""
    def __init__(self, in_channels: int = 3):
        super(MutualEncoder, self).__init__()
        
        # 6 convolution blocks as mentioned in paper
        self.blocks = nn.ModuleList()
        channels = [in_channels, 64, 128, 256, 512, 512, 512]
        
        for i in range(6):
            block = nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, 2, 1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            )
            self.blocks.append(block)
            
    def forward(self, x):
        features = [x]  # F_skip0 = I0
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
            
        return features


class LocalRetouchingBranch(nn.Module):
    """Local Retouching Branch (LRB) with Local Attentive Modules"""
    def __init__(self, feature_channels: List[int]):
        super(LocalRetouchingBranch, self).__init__()
        
        # Decoder path with skip connections
        self.up_convs = nn.ModuleList()
        self.lam_convs = nn.ModuleList()
        
        # Build decoder - simplified to avoid complex channel calculations
        decoder_channels = [512, 256, 128, 64, 32]
        
        for i in range(len(decoder_channels)-1):
            # Upsampling
            self.up_convs.append(nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], 4, 2, 1))
            
            # Calculate input channels for LAM
            if i < len(feature_channels) - 2:  # If we have skip connections
                skip_channels = feature_channels[-(i+2)]  # Corresponding skip feature
                in_ch = decoder_channels[i+1] + skip_channels + 1  # +1 for mask
            else:
                in_ch = decoder_channels[i+1] + 1  # Just upsampled + mask
                
            self.lam_convs.append(LocalAttentiveModule(in_ch, decoder_channels[i+1]))
            
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], 3, 3, 1, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, features, mask):
        x = features[-1]  # Start from deepest feature (512 channels)
        
        for i, (up_conv, lam_conv) in enumerate(zip(self.up_convs, self.lam_convs)):
            # Upsample
            x = up_conv(x)
            
            # Prepare inputs for LAM
            inputs_for_lam = [x]
            
            # Add skip connection if available
            skip_idx = len(features) - 2 - i
            if skip_idx >= 0 and skip_idx < len(features):
                skip = features[skip_idx]
                # Resize skip to match current feature size
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                inputs_for_lam.append(skip)
            
            # Add mask
            mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
            inputs_for_lam.append(mask_resized)
            
            # Concatenate all inputs
            x = torch.cat(inputs_for_lam, dim=1)
            
            # Apply LAM
            x = lam_conv(x)
            
        # Final output
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x


class ContextAwareLocalRetouchingLayer(nn.Module):
    """Context-aware Local Retouching Layer (LRL) with improved mask prediction"""
    def __init__(self, in_channels: int = 3):
        super(ContextAwareLocalRetouchingLayer, self).__init__()
        
        self.mutual_encoder = MutualEncoder(in_channels)
        
        # Get feature channels from encoder
        feature_channels = [3, 64, 128, 256, 512, 512, 512]
        
        # Use improved mask prediction branch
        self.mpb = MPB(64)
        self.lrb = LocalRetouchingBranch(feature_channels)

    def forward(self, x):
        # print(f"Input to LRL: {x.shape}")  # Should be [B, 3, 512, 512]
        # Store input size for final resize
        input_size = x.shape[2:]
        
        # Mutual encoding
        features = self.mutual_encoder(x)
        # print(f"features[1] shape: {features[1].shape}") 
        
        # Improved mask prediction
        mask = self.mpb(features[1])
        # print(f"Predicted mask shape: {mask.shape}")  
        # Local retouching
        retouched = self.lrb(features, mask)
        
        # Ensure output has same size as input
        if retouched.shape[2:] != input_size:
            # print('True')
            retouched = F.interpolate(retouched, size=input_size, mode='bilinear', align_corners=False)
        
        # Ensure mask has same size as input  
        # if mask.shape[2:] != input_size:
        #     mask = F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False)
        
        return retouched, mask




class AdaptiveBlendModule(nn.Module):
    """Adaptive Blend Module (ABM) for blending images - Simplified version"""
    def __init__(self):
        super(AdaptiveBlendModule, self).__init__()
        # Simplified adaptive blending with learnable parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, I, B):
        """Apply adaptive blend: R = alpha * I + beta * B + (1-alpha-beta) * I*B"""
        # Ensure alpha and beta are in reasonable ranges
        alpha = torch.sigmoid(self.alpha)  # 0 to 1
        beta = torch.sigmoid(self.beta)    # 0 to 1
        gamma = 1.0 - alpha - beta
        
        result = alpha * I + beta * B + gamma * I * B
        return torch.clamp(result, -1, 1)  # Clamp to reasonable range


class ReverseAdaptiveBlendModule(nn.Module):
    """Reverse Adaptive Blend Module (R-ABM) - Simplified version"""
    def __init__(self, abm: AdaptiveBlendModule):
        super(ReverseAdaptiveBlendModule, self).__init__()
        self.abm = abm  # Share parameters with ABM
        
    def forward(self, I, R):
        """Extract blend layer: Simplified inverse operation"""
        # Simplified inverse - solve for B given I and R
        alpha = torch.sigmoid(self.abm.alpha)
        beta = torch.sigmoid(self.abm.beta)
        gamma = 1.0 - alpha - beta
        
        # From R = alpha * I + beta * B + gamma * I * B
        # Solve for B: B = (R - alpha * I) / (beta + gamma * I)
        numerator = R - alpha * I
        denominator = beta + gamma * I
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-6)
        B = numerator / denominator
        
        return torch.clamp(B, -1, 1)  # Clamp to reasonable range


class RefiningModule(nn.Module):
    """Refining Module for upsampling blend layers"""
    def __init__(self):
        super(RefiningModule, self).__init__()
        # Light-weight refining as mentioned: 3x3 conv with 16 and 3 filters
        self.conv1 = nn.Conv2d(6, 16, 3, 1, 1)  # 3 (blend) + 3 (high-freq)
        self.conv2 = nn.Conv2d(16, 3, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, blend_layer, high_freq_component):
        """Refine blend layer with high-frequency component"""
        # Upsample blend layer
        upsampled_blend = F.interpolate(blend_layer, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Resize high-freq component to match upsampled blend size
        if high_freq_component.shape[2:] != upsampled_blend.shape[2:]:
            high_freq_component = F.interpolate(high_freq_component, size=upsampled_blend.shape[2:], 
                                              mode='bilinear', align_corners=False)
        
        # Concatenate with high-frequency component
        x = torch.cat([upsampled_blend, high_freq_component], dim=1)
        
        # Refine
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        
        # Residual connection
        refined_blend = x + upsampled_blend
        
        return refined_blend


class LaplacianPyramid(nn.Module):
    """Laplacian Pyramid for extracting high-frequency components"""
    def __init__(self):
        super(LaplacianPyramid, self).__init__()
        
    def forward(self, image_pyramid):
        """Extract high-frequency components from image pyramid"""
        high_freq_pyramid = []
        
        for i in range(len(image_pyramid) - 1):
            # Downsample higher resolution image
            downsampled = F.interpolate(image_pyramid[i], size=image_pyramid[i+1].shape[2:], 
                                      mode='bilinear', align_corners=False)
            # High-frequency = original - upsampled_lower_res
            upsampled = F.interpolate(image_pyramid[i+1], size=image_pyramid[i].shape[2:],
                                    mode='bilinear', align_corners=False)
            high_freq = image_pyramid[i] - upsampled
            high_freq_pyramid.append(high_freq)
            
        return high_freq_pyramid


class AdaptiveBlendPyramidLayer(nn.Module):
    """Adaptive Blend Pyramid Layer (BPL) for multi-scale processing"""
    def __init__(self, pyramid_levels: int = 2):
        super(AdaptiveBlendPyramidLayer, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        self.abm = AdaptiveBlendModule()
        self.r_abm = ReverseAdaptiveBlendModule(self.abm)
        self.refining_modules = nn.ModuleList([RefiningModule() for _ in range(pyramid_levels)])
        self.laplacian_pyramid = LaplacianPyramid()
        
    def forward(self, image_pyramid, low_res_input, low_res_output):
        """
        Args:
            image_pyramid: List of images at different resolutions [I0, I1, I2, ...]
            low_res_input: Low resolution input (I_l)
            low_res_output: Low resolution retouched result (R_l)
        """
        # Extract high-frequency components
        high_freq_pyramid = self.laplacian_pyramid(image_pyramid)
        
        # Get initial blend layer from low-res input/output
        blend_layer = self.r_abm(low_res_input, low_res_output)
        
        # Progressively refine blend layer
        for i in range(self.pyramid_levels):
            if i < len(high_freq_pyramid):
                # Use high-freq components in reverse order (from low to high res)
                hf_idx = len(high_freq_pyramid) - 1 - i
                blend_layer = self.refining_modules[i](blend_layer, high_freq_pyramid[hf_idx])
            else:
                # If no more high-freq components, just upsample
                blend_layer = F.interpolate(blend_layer, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Apply final blend to original high-res image
        final_result = self.abm(image_pyramid[0], blend_layer)
        
        return final_result, blend_layer


class ABPN(nn.Module):
    """
    Improved ABPN with better mask prediction from Architecture 2
    Combines fast pyramid processing with accurate mask prediction
    """
    def __init__(self, in_channels: int = 3, pyramid_levels: int = 2):
        super(ABPN, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        self.lrl = ContextAwareLocalRetouchingLayer(in_channels)  # Now has improved MPB
        self.bpl = AdaptiveBlendPyramidLayer(pyramid_levels)
        
    def build_image_pyramid(self, image, levels):
        """Build image pyramid by progressive downsampling"""
        pyramid = [image]
        current = image
        
        for _ in range(levels):
            current = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid.append(current)
            
        return pyramid
        
    def forward(self, x):
        """
        Args:
            x: Input high-resolution image
        Returns:
            final_result: Retouched high-resolution image
            mask: Predicted mask for target region (now much better quality!)
        """
        # Build image pyramid
        image_pyramid = self.build_image_pyramid(x, self.pyramid_levels)
        
        # Apply LRL to lowest resolution image (now with improved mask prediction)
        low_res_input = image_pyramid[-1]  # I_l
        low_res_output, mask = self.lrl(low_res_input)  # R_l, M (improved mask!)
        
        # Apply BPL to expand to original resolution
        final_result, _ = self.bpl(image_pyramid, low_res_input, low_res_output)
        
        return final_result, mask


# Test the improved model
if __name__ == "__main__":
    model = ABPN(in_channels=3, pyramid_levels=2)
    x = torch.randn(1, 3, 1024, 1024)
    
    with torch.no_grad():
        output, mask = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Mask shape: {mask.shape}")
        print("✅ Improved ABPN with better mask prediction working!")