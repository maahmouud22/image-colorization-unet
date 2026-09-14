import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from unet import UNet 
from DataSet import ColorizationDataset
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

# تسريع العمليات الحسابية على كرت الـ RTX 4070
torch.backends.cudnn.benchmark = True

def visualize_result(model, dataset, device, epoch=0):
    model.eval()
    L, ab = dataset[0] 
    L_tensor = L.unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_ab = model(L_tensor).squeeze().cpu().float().numpy().transpose(1, 2, 0)
    
    L_orig = (L.squeeze().numpy() + 1.) * 50.
    ab_orig = ab.numpy().transpose(1, 2, 0) * 110.
    ab_pred = pred_ab * 110.
    
    def get_rgb(l_chan, ab_chan):
        lab = np.zeros((128, 128, 3))
        lab[:, :, 0] = l_chan
        lab[:, :, 1:] = ab_chan
        with np.errstate(invalid='ignore'):
            return (lab2rgb(lab) * 255).astype(np.uint8)

    plt.figure(figsize=(15, 5))
    imgs = [L_orig, get_rgb(L_orig, ab_pred), get_rgb(L_orig, ab_orig)]
    titles = ["Input", f"Overnight Result (Epoch {epoch+1})", "Target"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgs[i], cmap='gray' if i==0 else None)
        plt.axis('off')
    plt.savefig(f"overnight_epoch_{epoch+1}.png")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device} | Starting the Final Overnight Run...")

    IMG_DIR = r"C:\Users\user\Downloads\coral 10k\archive (2)\Corek-10k" 
    BATCH_SIZE = 32
    EPOCHS = 100 # بما أنك ستنام، سنعطيه 100 دورة إضافية مع Learning Rate منخفض جداً

    dataset = ColorizationDataset(image_dir=IMG_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    model = UNet().to(device)
    
    # ⬅️ تحميل أوزان الجزء الثاني
    model_path = "unet_color_10k_L1_AMP_Part2.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Loaded: {model_path}. Continuing to the master level!")

    criterion = nn.L1Loss()
    
    # ⬅️ Learning Rate منخفض (1e-4) لضمان عدم تخريب الأوزان أثناء النوم
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # ⬅️ Scheduler هادئ (patience=5) ليتدخل فقط عند الضرورة القصوى
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda')

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for L, ab in dataloader:
            L, ab = L.to(device), ab.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(L)
                loss = criterion(outputs, ab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        print(f"Night Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"unet_color_10k_Final_Overnight.pth")
            visualize_result(model, dataset, device, epoch)

    plt.plot(loss_history)
    plt.savefig("loss_final_overnight.png")
    print(" التدريب انتهى! .")

if __name__ == "__main__":
    main()