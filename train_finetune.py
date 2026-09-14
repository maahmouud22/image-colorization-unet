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

# تسريع العمليات الحسابية
torch.backends.cudnn.benchmark = True

def visualize_result(model, dataset, device, epoch=0):
    model.eval()
    L, ab = dataset[0] 
    L_tensor = L.unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # استخدام float() لتجنب خطأ OpenCV مع دقة 16-بت
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
    titles = ["Input (B&W)", f"Result (Epoch {epoch+1})", "Target"]
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgs[i], cmap='gray' if i==0 else None)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.savefig(f"finetune_100_epoch_{epoch+1}.png")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device} | Starting 75 Extra Fine-Tuning Epochs...")

    # 👇 تأكد من أن هذا المسار يؤشر لمجلد الـ 14,300 صورة
    IMG_DIR = r"C:\Users\user\Downloads\coral 10k\archive (2)\Corek-10k" 
    
    BATCH_SIZE = 32
    EPOCHS = 75 # 75 دورة إضافية لكسر حاجز الألوان الباهتة

    dataset = ColorizationDataset(image_dir=IMG_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    model = UNet().to(device)
    
    # ⬅️ تحميل الأوزان من التدريب الأخير (الـ 25 دورة)
    model_path = "unet_color_Finetuned.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ تم تحميل الأوزان من {model_path}. سنكمل لـ 75 دورة إضافية!")
    else:
        print(f"❌ خطأ: لم يتم العثور على ملف {model_path}. تأكد من وجوده في نفس المجلد.")
        return

    criterion = nn.L1Loss() 
    
    # نحافظ على سرعة التعلم الهادئة للحصول على ألوان أعمق بدون تخريب
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # رفعنا الـ patience قليلاً لأن 75 دورة تعطينا مساحة أكبر للتدريب براحة
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda')

    loss_history = []

    print(f"🚀 Training on {len(dataset)} images for {EPOCHS} Epochs...")
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
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Extra Fine-Tune Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {current_lr:.7f}")
        
        if (epoch + 1) % 5 == 0:
            # ⬅️ الحفظ باسم جديد ليكون إجمالي الدورات 100
            torch.save(model.state_dict(), f"unet_color_Finetuned_100_epochs.pth")
            visualize_result(model, dataset, device, epoch)

    plt.plot(loss_history)
    plt.title("Fine-Tuning Progress (75 Extra Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.savefig("loss_finetuned_100_epochs.png")
    plt.show()
    print("🏁 اكتمل التدريب! تم حفظ الأوزان النهائية باسم unet_color_Finetuned_100_epochs.pth")

if __name__ == "__main__":
    main()