import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from unet import UNet  # تأكد أن هذا يطابق اسم ملف المعمارية لديك
from skimage.color import rgb2lab, lab2rgb

def test_single_image(model_path, image_path, save_name="test_result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")
    
    # 1. التحقق من وجود ملف الموديل
    if not os.path.exists(model_path):
        print(f"❌ خطأ: ملف الموديل '{model_path}' غير موجود!")
        return

    # 2. تحميل الموديل
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. قراءة الصورة
    # نستخدم طريقة القراءة الآمنة لدعم المسارات العربية إذا وجدت
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"❌ خطأ: لم أتمكن من قراءة الصورة من المسار: {image_path}")
        return

    img_rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb_full.shape[:2]

    # 4. تجهيز قناة التفاصيل العالية الدقة (L_high_res)
    img_lab_full = rgb2lab(img_rgb_full).astype("float32")
    L_high_res = img_lab_full[:, :, 0] 

    # 5. تجهيز النسخة المصغرة للموديل (128x128)
    img_resized = cv2.resize(img_rgb_full, (128, 128))
    img_lab_low = rgb2lab(img_resized).astype("float32")
    L_low = img_lab_low[:, :, 0] / 50.0 - 1.0
    L_tensor = torch.from_numpy(L_low).unsqueeze(0).unsqueeze(0).to(device)

    # 6. التنبؤ بالألوان
    print("🎨 جاري تلوين الصورة...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'): # نستخدم AMP لأننا دربناه بها
            # ⬅️ التعديل هنا: إضافة .float() لتحويل الأرقام لـ 32-بت لكي يقبلها OpenCV
            pred_ab_low = model(L_tensor).squeeze().cpu().float().numpy().transpose(1, 2, 0)
    
    # 7. الدمج الذكي (Smart Merge)
    # تكبير الألوان الناتجة لتطابق حجم الصورة الأصلية
    ab_pred_high = cv2.resize(pred_ab_low * 110.0, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # تركيب الألوان مع التفاصيل الأصلية
    result_lab_high = np.zeros((h, w, 3))
    result_lab_high[:, :, 0] = L_high_res
    result_lab_high[:, :, 1:] = ab_pred_high
    
    # التحويل النهائي للعرض
    with np.errstate(invalid='ignore'):
        result_rgb_high = np.clip(lab2rgb(result_lab_high) * 255, 0, 255).astype(np.uint8)

    # 8. عرض النتيجة وحفظها
    plt.figure(figsize=(12, 6))
    
    # عرض الصورة كأبيض وأسود
    plt.subplot(1, 2, 1)
    plt.imshow(L_high_res, cmap='gray')
    plt.title("Input (Black & White)")
    plt.axis('off')
    
    # عرض الصورة الملونة
    plt.subplot(1, 2, 2)
    plt.imshow(result_rgb_high)
    plt.title("AI Colorization")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"✅ تمت العملية بنجاح! تم حفظ النتيجة باسم: {save_name}")
    plt.show()

if __name__ == "__main__":
    # --- إعداداتك ---
    # اسم ملف الأوزان اللي خلص 50 دورة
    MODEL_FILE = "unet_color_10k_Final_Overnight.pth" 
    
    # 👇 ضع مسار الصورة اللي بدك تفحصها هنا 👇
    TEST_IMAGE_PATH = r"C:\Users\user\Downloads\pexels-patryk-nowicki-2161404750-37459980.jpg"
    test_single_image(MODEL_FILE, TEST_IMAGE_PATH)