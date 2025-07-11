import argparse
import os
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from model import PartBasedTransformer  # Import the modified model

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='/home/liuhaoan/ADTS/AstroTracNet/astro_sort/astro/Market-1501', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=3, type=int)
parser.add_argument("--lr", default=0.0002, type=float)  # Smaller learning rate
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--epochs', default=18, type=int)  # More training epochs
parser.add_argument('--num-parts', default=3, type=int)  # Parameterize number of parts
parser.add_argument('--visualize-attention', action='store_true')  # Add visualization switch
args = parser.parse_args()

# Create runs directory if it does not exist
runs_dir = "checkpoint"
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

# Automatically create experiment directory (exp1, exp2, ...) under runs directory
existing_exps = glob.glob(os.path.join(runs_dir, "exp*"))
exp_nums = [int(os.path.basename(exp).replace("exp", "")) for exp in existing_exps 
            if os.path.basename(exp).replace("exp", "").isdigit()]
next_exp_num = 1 if not exp_nums else max(exp_nums) + 1
exp_dir = os.path.join(runs_dir, f"exp{next_exp_num}")

# Create experiment directory and its checkpoint subdirectory
exp_checkpoint_dir = os.path.join(exp_dir, "checkpoint")
vis_dir = os.path.join(exp_dir, "attention_vis")  # Add attention visualization directory
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(exp_checkpoint_dir):
    os.makedirs(exp_checkpoint_dir)
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

print(f"Experiment will be saved in directory: {exp_dir}")

# device
device = "cuda:{}".format(
    args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# Enhanced data loading and augmentation
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")

# Stronger data augmentation
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=10),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    torchvision.transforms.RandomErasing(p=0.5)
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64,  # Smaller batch size
    shuffle=True,
    num_workers=8  # Use multi-threaded loading
)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64,
    shuffle=True,
    num_workers=8
)

num_classes = max(len(trainloader.dataset.classes),
                  len(testloader.dataset.classes))

# Create PartBasedTransformer model, now using attention-guided partitioning
start_epoch = 0
net = PartBasedTransformer(
    img_size=(128, 64),
    embed_dim=128,
    num_parts=args.num_parts,  # Use command line parameter
    depth=2,
    num_heads=4,
    num_classes=num_classes,
    reid=False  # False during training
)

if args.resume:
    checkpoint_path = "./checkpoint/ckpt.t7"
    assert os.path.isfile(checkpoint_path), f"Error: no checkpoint file found at {checkpoint_path}!"
    print(f'Loading from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# Use AdamW optimizer and cosine annealing learning rate scheduler
optimizer = torch.optim.AdamW(
    net.parameters(), 
    lr=args.lr, 
    weight_decay=1e-4  # Moderate weight decay
)

# Cosine annealing learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=args.epochs
)

# Use cross-entropy loss function
criterion = torch.nn.CrossEntropyLoss()
best_acc = 0.

# Add attention regularization loss to encourage different parts to focus on different regions
def attention_diversity_loss(attention_maps):
    """Compute attention diversity loss to encourage different parts to focus on different regions"""
    batch_size, num_parts, h, w = attention_maps.size()
    attention_flat = attention_maps.view(batch_size, num_parts, -1)
    
    # Compute similarity matrix between parts
    similarity = torch.bmm(attention_flat, attention_flat.transpose(1, 2))
    
    # Diagonal elements should be close to 1 (self-similarity), off-diagonal should be close to 0 (different parts do not overlap)
    # Create target matrix (diagonal is 1, others are 0)
    target = torch.eye(num_parts, device=attention_maps.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Compute mean squared error between similarity matrix and target matrix
    diversity_loss = torch.nn.functional.mse_loss(similarity, target)
    
    return diversity_loss

# Attention map visualization function
def visualize_attention_maps(images, attention_maps, epoch, batch_idx, save_dir):
    """Save attention map visualization results"""
    if batch_idx % 50 != 0:  # Visualize every 50 batches
        return
        
    # Only process the first image in the batch
    image = images[0].cpu().permute(1, 2, 0).numpy()
    # De-normalize
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    # Create visualization figure
    num_parts = attention_maps.size(1)
    fig, axes = plt.subplots(1, num_parts + 1, figsize=(4 * (num_parts + 1), 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention maps
    for i in range(num_parts):
        att = attention_maps[0, i].detach().cpu().numpy()
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(att, alpha=0.5, cmap='jet')
        axes[i + 1].set_title(f'Part {i+1} Attention')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'attention_e{epoch}_b{batch_idx}.png')
    plt.savefig(save_path)
    plt.close(fig)

# Training function
def train(epoch):
    print("\nEpoch : %d" % (epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get base features and attention maps
        feat = net.features(inputs)
        part_attention = net.part_locator(feat)
        part_attention = torch.softmax(part_attention.view(inputs.size(0), net.num_parts, -1), dim=2)
        part_attention = part_attention.view(inputs.size(0), net.num_parts, feat.size(2), feat.size(3))
        
        # Forward pass
        outputs = net(inputs)
        
        # Since the part-based model returns multiple classification results, special handling is needed
        if isinstance(outputs, list):
            # Compute loss for each part classifier and the fusion classifier
            loss = 0
            for part_pred in outputs[:-1]:  # All classifier predictions, last one is fusion prediction
                loss += criterion(part_pred, labels)
            
            # Increase weight for fusion classifier loss
            fusion_pred = outputs[-1]
            loss += 2.0 * criterion(fusion_pred, labels)
            
            # Add attention diversity loss
            diversity_loss = attention_diversity_loss(part_attention)
            loss += 0.1 * diversity_loss  # The weight coefficient can be adjusted
            
            # Record fusion classifier loss separately
            fusion_loss = criterion(fusion_pred, labels)

            # Use fusion prediction for accuracy calculation
            pred = fusion_pred.max(dim=1)[1]
        else:
            # Single prediction case
            loss = criterion(outputs, labels)
            fusion_loss = loss
            pred = outputs.max(dim=1)[1]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualize attention maps (if enabled)
        if args.visualize_attention and idx % 50 == 0:  # Visualize every 50 batches
            # Upsample attention maps to original image size for visualization
            upsampled_attention = torch.nn.functional.interpolate(
                part_attention, 
                size=(inputs.size(2), inputs.size(3)), 
                mode='bilinear',
                align_corners=False
            )
            visualize_attention_maps(inputs, upsampled_attention, epoch, idx, vis_dir)

        # Accumulate loss and accuracy
        training_loss += fusion_loss.item()
        train_loss += fusion_loss.item()
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        # Print progress
        if (idx+1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss /
                interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss/len(trainloader), 1. - correct/total

# Test function
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            
            # Also handle multi-part outputs
            if isinstance(outputs, list):
                # Use fusion classifier output for loss calculation
                loss = criterion(outputs[-1], labels)
                pred = outputs[-1].max(dim=1)[1]
            else:
                loss = criterion(outputs, labels)
                pred = outputs.max(dim=1)[1]

            test_loss += loss.item()
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100.*(idx+1)/len(testloader), end-start, test_loss /
            len(testloader), correct, total, 100.*correct/total
        ))

    # Save the best model
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        best_ckpt_path = os.path.join(exp_checkpoint_dir, "ckpt.t7")
        print(f"Saving parameters to {best_ckpt_path}")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(checkpoint, best_ckpt_path)
        
        # Visualize attention maps of the best model on the test set
        if args.visualize_attention:
            visualize_best_attention(inputs, epoch)

    return test_loss/len(testloader), 1. - correct/total

# Visualize attention maps of the best model
def visualize_best_attention(test_images, epoch):
    """Visualize the attention allocation of the best model for several images in the test set"""
    # Select the first 8 images
    sample_images = test_images[:8].to(device)
    
    # Get base features and attention maps
    with torch.no_grad():
        feat = net.features(sample_images)
        part_attention = net.part_locator(feat)
        part_attention = torch.softmax(part_attention.view(sample_images.size(0), net.num_parts, -1), dim=2)
        part_attention = part_attention.view(sample_images.size(0), net.num_parts, feat.size(2), feat.size(3))
        
        # Upsample to original image size
        upsampled_attention = torch.nn.functional.interpolate(
            part_attention, 
            size=(sample_images.size(2), sample_images.size(3)), 
            mode='bilinear',
            align_corners=False
        )
    
    # Create visualization for each sample
    for i in range(min(8, sample_images.size(0))):
        image = sample_images[i].cpu().permute(1, 2, 0).numpy()
        # De-normalize
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        
        # Create visualization figure
        fig, axes = plt.subplots(1, net.num_parts + 1, figsize=(4 * (net.num_parts + 1), 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Attention maps
        for j in range(net.num_parts):
            att = upsampled_attention[i, j].cpu().numpy()
            axes[j + 1].imshow(image)
            axes[j + 1].imshow(att, alpha=0.5, cmap='jet')
            axes[j + 1].set_title(f'Part {j+1}')
            axes[j + 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f'best_attention_e{epoch}_sample{i}.png')
        plt.savefig(save_path)
        plt.close(fig)

# Plotting function
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    
    # Save curve plot to experiment directory
    plot_path = os.path.join(exp_dir, "train.jpg")
    fig.savefig(plot_path)

# Main function
def main():
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        
        # Update learning rate
        scheduler.step()
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Periodically save checkpoints
        if epoch % 5 == 0 or current_lr < 1e-5:
            print(f"Saving periodic checkpoint, epoch {epoch}")
            checkpoint = {
                'net_dict': net.state_dict(),
                'acc': 100. * (1 - test_err),
                'epoch': epoch,
            }
            epoch_ckpt_path = os.path.join(exp_checkpoint_dir, f"ckpt_epoch_{epoch}.t7")
            torch.save(checkpoint, epoch_ckpt_path)

if __name__ == '__main__':
    main()
