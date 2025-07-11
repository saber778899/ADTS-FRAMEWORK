import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PartBasedTransformer(nn.Module):
    def __init__(self, img_size=(128, 64), embed_dim=128, num_parts=3, 
                 depth=2, num_heads=4, num_classes=751, reid=False):
        super().__init__()
        self.reid = reid
        self.num_parts = num_parts
        
        # Basic feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x64x32
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x32x16
            
            nn.Conv2d(128, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True)  # 256x32x16
        )
        
        # Add part locator network - attention-guided adaptive partitioning
        self.part_locator = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.Conv2d(embed_dim, num_parts, 1)  # Each channel corresponds to an attention map for a part
        )
        
        # Create a separate Transformer for each part
        self.part_transformers = nn.ModuleList()
        for _ in range(num_parts):
            encoder_layer = TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            )
            self.part_transformers.append(
                TransformerEncoder(encoder_layer, num_layers=depth)
            )
        
        # Global feature Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.global_transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classifiers
        self.part_classifiers = nn.ModuleList()
        for _ in range(num_parts):
            self.part_classifiers.append(nn.Linear(embed_dim, num_classes))
        
        self.global_classifier = nn.Linear(embed_dim, num_classes)
        
        # Feature fusion layer
        self.fusion = nn.Linear(embed_dim * (num_parts + 1), num_classes)
        
        # Add Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Basic feature extraction
        feat = self.features(x)  # B x C x H x W
        
        # Generate part attention maps
        part_attention = self.part_locator(feat)  # B x num_parts x H x W
        
        # Convert attention maps to spatial probability distributions (for each part, the sum of weights over all positions is 1)
        part_attention = torch.softmax(part_attention.view(B, self.num_parts, -1), dim=2)
        part_attention = part_attention.view(B, self.num_parts, feat.size(2), feat.size(3))
        
        # Use attention maps to extract features for each part
        part_features = []
        
        for i in range(self.num_parts):
            # Apply attention weights to feature map
            weighted_feat = feat * part_attention[:, i:i+1, :, :]  # B x C x H x W
            
            # Convert to tokens and pass through Transformer
            part_tokens = weighted_feat.flatten(2).transpose(1, 2)  # B x (H*W) x C
            part_tokens = self.part_transformers[i](part_tokens)
            
            # Global pooling to obtain the feature vector for this part
            part_vector = self.dropout(part_tokens.mean(dim=1))  # B x C
            part_features.append(part_vector)
        
        # Global features
        global_tokens = feat.flatten(2).transpose(1, 2)  # B x (H*W) x C
        global_tokens = self.global_transformer(global_tokens)
        global_vector = global_tokens.mean(dim=1)  # B x C
        part_features.append(global_vector)
        
        # Feature fusion
        all_features = torch.cat(part_features, dim=1)  # B x (num_parts+1)*C
        
        # reid mode
        if self.reid:
            return all_features.div(all_features.norm(p=2, dim=1, keepdim=True))
        
        # Compute predictions for each part and global
        part_preds = [classifier(feat) for classifier, feat in 
                      zip(self.part_classifiers, part_features[:-1])]
        global_pred = self.global_classifier(global_vector)
        
        # Fuse all features for final prediction
        fusion_pred = self.fusion(all_features)
        
        # During training, return all predictions (multi-task learning); during testing, return only the fusion prediction
        if self.training:
            return part_preds + [global_pred, fusion_pred]
        else:
            return fusion_pred

    def visualize_attention(self, x):
        """
        Used to visualize the attention maps for each part
        Returns: original image and corresponding list of attention maps
        """
        B = x.shape[0]
        
        # Basic feature extraction
        feat = self.features(x)  # B x C x H x W
        
        # Generate part attention maps
        part_attention = self.part_locator(feat)  # B x num_parts x H x W
        part_attention = torch.softmax(part_attention.view(B, self.num_parts, -1), dim=2)
        part_attention = part_attention.view(B, self.num_parts, feat.size(2), feat.size(3))
        
        # Upsample attention maps to original image size
        attention_maps = []
        for i in range(self.num_parts):
            attention_map = nn.functional.interpolate(
                part_attention[:, i:i+1, :, :],
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
            attention_maps.append(attention_map)
        
        return x, attention_maps