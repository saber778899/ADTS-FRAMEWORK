import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt

from .model import PartBasedTransformer


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        # Use attention-guided Transformer model
        self.net = PartBasedTransformer(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        
        # Load pre-trained weights
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        if 'net_dict' in state_dict:
            state_dict = state_dict['net_dict']
        self.net.load_state_dict(state_dict)
        
        logger = logging.getLogger("root.tracker")
        logger.info("Loading Attention-Guided Transformer weights from {}... Done!".format(model_path))
        
        self.net.to(self.device)
        self.net.eval()  # Set to evaluation mode
        self.size = (64, 128)  # Keep the original size
        
        # Image normalization consistent with previous settings
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        Preprocess images:
        1. Convert to float in range 0-1
        2. Resize to (64, 128)
        3. Normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        """
        Extract feature vectors
        """
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
    
    def extract_with_attention(self, im_crop):
        """
        Extract features and return attention maps for visualization
        """
        im_batch = self._preprocess([im_crop])
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            image, attention_maps = self.net.visualize_attention(im_batch)
        
        # Convert to numpy array
        image = image[0].cpu().permute(1, 2, 0).numpy()
        # Denormalize
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        attention_numpy = [att[0, 0].cpu().numpy() for att in attention_maps]
        
        return image, attention_numpy
    
    def visualize_attention(self, im_crop, save_path=None):
        """
        Visualize attention maps for each part
        """
        image, attention_maps = self.extract_with_attention(im_crop)
        
        # Create image grid
        n_parts = len(attention_maps)
        plt.figure(figsize=(12, 4))
        
        # Show original image
        plt.subplot(1, n_parts+1, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show attention map for each part
        for i, att_map in enumerate(attention_maps):
            plt.subplot(1, n_parts+1, i+2)
            plt.imshow(att_map, cmap='jet')
            plt.title(f'Part {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
    