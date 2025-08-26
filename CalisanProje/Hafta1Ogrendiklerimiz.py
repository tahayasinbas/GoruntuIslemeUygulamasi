import cv2
import numpy as np
import os
from tkinter import filedialog

def resim_ac():
    """
    Kullanıcının bilgisayarından bir resim dosyası seçmesini sağlar ve dosya yolunu döndürür
    
    Returns:
        str: Seçilen dosyanın yolu, dosya seçilmediyse None
    """
    try:
        # Başlangıç dizini olarak masaüstü veya belgeler klasörünü dene
        initial_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")  # Kullanıcı klasörü
        
        # Dosya seçme dialogunu aç
        dosya_yolu = filedialog.askopenfilename(
            title="Resim Dosyası Seç",
            initialdir=initial_dir,
            filetypes=[
                ("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("JPEG Dosyaları", "*.jpg *.jpeg"),
                ("PNG Dosyaları", "*.png"),
                ("BMP Dosyaları", "*.bmp"),
                ("TIFF Dosyaları", "*.tif *.tiff"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        return dosya_yolu
            
    except Exception as e:
        print(f"Hata: Dosya dialog penceresi açılırken bir hata oluştu: {str(e)}")
        return None

def goruntu_oku(dosya_yolu):
    """
    Belirtilen dosya yolundaki resmi yükler ve döndürür
    
    Args:
        dosya_yolu (str): Yüklenecek görüntünün dosya yolu
        
    Returns:
        numpy.ndarray: Yüklenen görüntü, hata durumunda None
    """
    try:
        # Resmi oku
        img = cv2.imread(dosya_yolu)
        if img is None:
            print("Hata: Resim dosyası yüklenemedi! Dosya formatı desteklenmiyor olabilir.")
            return None
        
        return img
    
    except Exception as e:
        print(f"Hata: Resim yüklenirken bir hata oluştu: {str(e)}")
        return None

def goruntu_kaydet(img, kalite=95):
    """
    Görüntüyü kullanıcının seçtiği konuma kaydeder
    
    Args:
        img (numpy.ndarray): Kaydedilecek görüntü
        kalite (int, optional): JPEG kalite değeri (0-100). Varsayılan 95.
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    if img is None:
        print("Uyarı: Kaydedilecek bir görüntü yok!")
        return False
    
    try:
        # Dosya kaydetme dialogunu aç
        dosya_yolu = filedialog.asksaveasfilename(
            title="Görüntüyü Kaydet",
            initialdir=os.path.join(os.path.expanduser("~"), "Desktop"),
            filetypes=[
                ("JPEG Dosyaları", "*.jpg"),
                ("PNG Dosyaları", "*.png"),
                ("BMP Dosyaları", "*.bmp"),
                ("TIFF Dosyaları", "*.tif")
            ],
            defaultextension=".jpg"
        )
        
        if not dosya_yolu:
            return False
        
        # Dosya uzantısını kontrol et
        _, ext = os.path.splitext(dosya_yolu)
        ext = ext.lower()
        
        # JPEG için kalite ayarı, diğerleri için varsayılan parametreler
        if ext == ".jpg" or ext == ".jpeg":
            cv2.imwrite(dosya_yolu, img, [cv2.IMWRITE_JPEG_QUALITY, kalite])
        elif ext == ".png":
            cv2.imwrite(dosya_yolu, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:  # BMP, TIFF ve diğerleri
            cv2.imwrite(dosya_yolu, img)
        
        print(f"Bilgi: Görüntü başarıyla kaydedildi: {dosya_yolu}")
        return True
        
    except Exception as e:
        print(f"Hata: Görüntü kaydedilirken bir hata oluştu: {str(e)}")
        return False

def griye_cevir(img):
    """
    Görüntüyü gri tonlamalı hale getirir
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        
    Returns:
        numpy.ndarray: Gri tonlamalı görüntü, hata durumunda None
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
        
    try:
        # BGR'dan gri tonlamaya dönüştür
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
        
    except Exception as e:
        print(f"Hata: Gri tonlama işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def negatif_al(img):
    """
    Görüntünün negatifini alır
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        
    Returns:
        numpy.ndarray: Negatif görüntü, hata durumunda None
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
        
    try:
        # 255'ten görüntüyü çıkararak negatif al
        negative = 255 - img
        return negative
        
    except Exception as e:
        print(f"Hata: Negatif işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def kanallara_ayir(img):
    """
    Görüntüyü R, G, B kanallarına ayırır
    
    Args:
        img (numpy.ndarray): İşlenecek BGR görüntüsü
        
    Returns:
        tuple: (B kanalı, G kanalı, R kanalı) olarak ayrılmış görüntüler
               Hata durumunda (None, None, None)
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None, None, None
        
    try:
        # Görüntüyü ayrı kanallara ayır
        b, g, r = cv2.split(img)
        return b, g, r
        
    except Exception as e:
        print(f"Hata: Kanal ayırma işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None, None, None

def kanali_goster(img, kanal):
    """
    Belirli bir renk kanalını görselleştirir (B, G veya R)
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kanal (str): 'blue', 'green' veya 'red' olarak kanal seçimi
        
    Returns:
        numpy.ndarray: Belirtilen kanalı içeren görüntü, hata durumunda None
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
        
    try:
        # Görüntüyü ayrı kanallara ayır
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)
        
        if kanal.lower() == 'blue' or kanal == 0:
            result = cv2.merge([b, zeros, zeros])
        elif kanal.lower() == 'green' or kanal == 1:
            result = cv2.merge([zeros, g, zeros])
        elif kanal.lower() == 'red' or kanal == 2:
            result = cv2.merge([zeros, zeros, r])
        else:
            print("Hata: Geçersiz kanal seçimi! (blue, green veya red olmalı)")
            return None
            
        return result
        
    except Exception as e:
        print(f"Hata: Renk kanalı işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def goruntu_goster(img, baslik="Görüntü", bekle=True):
    """
    Görüntüyü ekranda gösterir
    
    Args:
        img (numpy.ndarray): Gösterilecek görüntü
        baslik (str, optional): Pencere başlığı. Varsayılan "Görüntü".
        bekle (bool, optional): True ise bir tuşa basılana kadar bekler, 
                               False ise 1ms bekleyip devam eder. Varsayılan True.
    """
    if img is None:
        print("Hata: Gösterilecek bir görüntü yok!")
        return
        
    try:
        cv2.imshow(baslik, img)
        
        if bekle:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
            
    except Exception as e:
        print(f"Hata: Görüntü gösterilirken bir hata oluştu: {str(e)}")