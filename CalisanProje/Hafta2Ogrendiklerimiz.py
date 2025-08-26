import cv2
import numpy as np
import matplotlib.pyplot as plt

def parlaklik_ayarla(img, deger):
    """
    Görüntünün parlaklığını artırır veya azaltır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        deger (int): Pozitif değerler parlaklığı artırır, negatif değerler azaltır
        
    Returns:
        numpy.ndarray: Parlaklığı ayarlanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Görüntü kopyasını oluştur
        sonuc = img.copy()
        
        # Görüntü gri tonlamalı ise
        if len(img.shape) == 2:
            rows, cols = img.shape
            for i in range(rows):
                for j in range(cols):
                    piksel = int(img[i, j]) + deger
                    # Sınırları kontrol et
                    if piksel > 255:
                        sonuc[i, j] = 255
                    elif piksel < 0:
                        sonuc[i, j] = 0
                    else:
                        sonuc[i, j] = piksel
        
        # Görüntü renkli ise
        elif len(img.shape) == 3:
            rows, cols, channels = img.shape
            for i in range(rows):
                for j in range(cols):
                    for k in range(channels):
                        piksel = int(img[i, j, k]) + deger
                        # Sınırları kontrol et
                        if piksel > 255:
                            sonuc[i, j, k] = 255
                        elif piksel < 0:
                            sonuc[i, j, k] = 0
                        else:
                            sonuc[i, j, k] = piksel
        
        return sonuc
        
    except Exception as e:
        print(f"Hata: Parlaklık ayarlanırken bir hata oluştu: {str(e)}")
        return None

def parlaklik_ayarla_hizli(img, deger):
    """
    Görüntünün parlaklığını hızlı bir şekilde artırır veya azaltır (vektörel işlemlerle).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        deger (int): Pozitif değerler parlaklığı artırır, negatif değerler azaltır
        
    Returns:
        numpy.ndarray: Parlaklığı ayarlanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # NumPy vektörel işlemleri kullanarak parlaklık ayarı
        sonuc = np.clip(img.astype(np.int16) + deger, 0, 255).astype(np.uint8)
        return sonuc
        
    except Exception as e:
        print(f"Hata: Parlaklık ayarlanırken bir hata oluştu: {str(e)}")
        return None

def esikleme(img, esik_degeri, max_deger=255):
    """
    Görüntüye eşikleme işlemi uygular (piksel değeri eşik değerinden büyükse 
    max_deger, değilse 0 yapar).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        esik_degeri (int): Eşik değeri
        max_deger (int, optional): Eşik üstündeki piksellere atanacak değer. Varsayılan 255.
        
    Returns:
        numpy.ndarray: Eşiklenmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    # Görüntü renkli ise gri tonlamaya çevir
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    try:
        # Manuel eşikleme
        rows, cols = gray.shape
        sonuc = np.zeros_like(gray)
        
        for i in range(rows):
            for j in range(cols):
                if gray[i, j] >= esik_degeri:
                    sonuc[i, j] = max_deger
                else:
                    sonuc[i, j] = 0
                    
        return sonuc
        
    except Exception as e:
        print(f"Hata: Eşikleme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def esikleme_hizli(img, esik_degeri, max_deger=255):
    """
    Görüntüye eşikleme işlemi uygular (OpenCV kullanarak).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        esik_degeri (int): Eşik değeri
        max_deger (int, optional): Eşik üstündeki piksellere atanacak değer. Varsayılan 255.
        
    Returns:
        numpy.ndarray: Eşiklenmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    # Görüntü renkli ise gri tonlamaya çevir
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    try:
        # OpenCV'nin yerleşik fonksiyonunu kullanarak eşikleme
        _, sonuc = cv2.threshold(gray, esik_degeri, max_deger, cv2.THRESH_BINARY)
        return sonuc
        
    except Exception as e:
        print(f"Hata: Eşikleme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def adaptif_esikleme(img, max_deger=255, adaptif_yontem=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    esik_tipi=cv2.THRESH_BINARY, blok_boyutu=11, c=2):
    """
    Görüntüye adaptif eşikleme uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        max_deger (int, optional): Eşik üstündeki piksellere atanacak değer. Varsayılan 255.
        adaptif_yontem: cv2.ADAPTIVE_THRESH_MEAN_C veya cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        esik_tipi: cv2.THRESH_BINARY veya cv2.THRESH_BINARY_INV
        blok_boyutu (int): Adaptif eşikleme için komşuluk bloğunun boyutu (tek sayı olmalı)
        c (int): Hesaplanan eşik değerinden çıkarılan sabit
        
    Returns:
        numpy.ndarray: Adaptif eşiklenmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    # Görüntü renkli ise gri tonlamaya çevir
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    try:
        # OpenCV'nin yerleşik fonksiyonunu kullanarak adaptif eşikleme
        sonuc = cv2.adaptiveThreshold(gray, max_deger, adaptif_yontem, 
                                      esik_tipi, blok_boyutu, c)
        return sonuc
        
    except Exception as e:
        print(f"Hata: Adaptif eşikleme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def histogram_hesapla(img, goster=False):
    """
    Görüntünün histogramını hesaplar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        goster (bool, optional): True ise histogramı gösterir. Varsayılan False.
        
    Returns:
        numpy.ndarray: Histogram verileri
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Görüntü renkli ise
        if len(img.shape) == 3:
            # Her kanal için ayrı histogram hesapla
            color = ('b', 'g', 'r')
            hist_data = []
            
            for i, col in enumerate(color):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                hist_data.append(hist)
                
                if goster:
                    plt.plot(hist, color=col)
            
            if goster:
                plt.title("RGB Histogram")
                plt.xlabel("Piksel Değeri")
                plt.ylabel("Frekans")
                plt.xlim([0, 256])
                plt.show()
                
            return hist_data
        
        # Görüntü gri tonlamalı ise
        else:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            
            if goster:
                plt.figure()
                plt.title("Gri Seviye Histogram")
                plt.xlabel("Piksel Değeri")
                plt.ylabel("Frekans")
                plt.plot(hist)
                plt.xlim([0, 256])
                plt.show()
                
            return hist
            
    except Exception as e:
        print(f"Hata: Histogram hesaplanırken bir hata oluştu: {str(e)}")
        return None

def histogram_esitleme(img, goster=False):
    """
    Görüntünün histogramını eşitler.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        goster (bool, optional): True ise orijinal ve eşitlenmiş görüntü histogramlarını gösterir.
                                Varsayılan False.
        
    Returns:
        numpy.ndarray: Histogram eşitlenmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Görüntü renkli ise
        if len(img.shape) == 3:
            # BGR görüntüyü YCrCb renk uzayına dönüştür
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            
            # Y kanalını (parlaklık) ayır
            y, cr, cb = cv2.split(ycrcb)
            
            # Sadece Y kanalında histogram eşitleme yap
            y_eq = cv2.equalizeHist(y)
            
            # Kanalları birleştir
            ycrcb_eq = cv2.merge([y_eq, cr, cb])
            
            # YCrCb'den BGR'a geri dönüştür
            sonuc = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
            
            if goster:
                # Orijinal ve eşitlenmiş histogramları göster
                plt.figure(figsize=(12, 5))
                
                # BGR histogramları
                for i, col in enumerate(['b', 'g', 'r']):
                    hist_orig = cv2.calcHist([img], [i], None, [256], [0, 256])
                    hist_eq = cv2.calcHist([sonuc], [i], None, [256], [0, 256])
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(hist_orig, color=col)
                    plt.title("Orijinal Histogram")
                    plt.xlabel("Piksel Değeri")
                    plt.ylabel("Frekans")
                    plt.xlim([0, 256])
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(hist_eq, color=col)
                    plt.title("Eşitlenmiş Histogram")
                    plt.xlabel("Piksel Değeri")
                    plt.ylabel("Frekans")
                    plt.xlim([0, 256])
                
                plt.tight_layout()
                plt.show()
            
            return sonuc
        
        # Görüntü gri tonlamalı ise
        else:
            # Histogram eşitleme
            sonuc = cv2.equalizeHist(img)
            
            if goster:
                # Orijinal ve eşitlenmiş histogramları göster
                hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_eq = cv2.calcHist([sonuc], [0], None, [256], [0, 256])
                
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.title("Orijinal Histogram")
                plt.xlabel("Piksel Değeri")
                plt.ylabel("Frekans")
                plt.plot(hist_orig, color='black')
                plt.xlim([0, 256])
                
                plt.subplot(1, 2, 2)
                plt.title("Eşitlenmiş Histogram")
                plt.xlabel("Piksel Değeri")
                plt.ylabel("Frekans")
                plt.plot(hist_eq, color='black')
                plt.xlim([0, 256])
                
                plt.tight_layout()
                plt.show()
            
            return sonuc
            
    except Exception as e:
        print(f"Hata: Histogram eşitleme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def kontrast_germe(img, goster=False):
    """
    Doğrusal kontrast germe uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        goster (bool, optional): True ise orijinal ve kontrast gerilmiş görüntüleri gösterir.
                                Varsayılan False.
        
    Returns:
        numpy.ndarray: Kontrast gerilmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Görüntü renkli ise her kanalı ayrı ayrı işle
        if len(img.shape) == 3:
            # Kanalları ayır
            b, g, r = cv2.split(img)
            
            # Her kanal için kontrast germe
            b_stretched = kontrast_germe_kanali(b)
            g_stretched = kontrast_germe_kanali(g)
            r_stretched = kontrast_germe_kanali(r)
            
            # Kanalları birleştir
            sonuc = cv2.merge([b_stretched, g_stretched, r_stretched])
            
        else:
            # Gri tonlamalı görüntü için kontrast germe
            sonuc = kontrast_germe_kanali(img)
        
        if goster:
            plt.figure(figsize=(10, 5))
            
            if len(img.shape) == 3:
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title('Orijinal')
                
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
                plt.title('Kontrast Gerilmiş')
            else:
                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.title('Orijinal')
                
                plt.subplot(1, 2, 2)
                plt.imshow(sonuc, cmap='gray')
                plt.title('Kontrast Gerilmiş')
            
            plt.tight_layout()
            plt.show()
        
        return sonuc
        
    except Exception as e:
        print(f"Hata: Kontrast germe işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def kontrast_germe_kanali(kanal):
    """
    Görüntü kanalına doğrusal kontrast germe uygular.
    
    Args:
        kanal (numpy.ndarray): İşlenecek tek kanallı görüntü
        
    Returns:
        numpy.ndarray: Kontrast gerilmiş kanal
    """
    # Minimum ve maksimum piksel değerlerini bul
    min_val = np.min(kanal)
    max_val = np.max(kanal)
    
    # Kontrast germe formülüyle yeni görüntüyü hesapla
    # (x - min) / (max - min) * 255
    if max_val > min_val:
        return ((kanal.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        return kanal  # min ve max aynıysa değişiklik yapmadan döndür

def kontrast_ayarla(img, alpha, beta=0):
    """
    Görüntüye kontrast (alpha) ve parlaklık (beta) ayarı uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        alpha (float): Kontrast faktörü (1.0 = orijinal kontrast)
        beta (int, optional): Parlaklık değeri. Varsayılan 0.
        
    Returns:
        numpy.ndarray: Kontrast ve parlaklığı ayarlanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Formül: g(x,y) = alpha * f(x,y) + beta
        sonuc = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return sonuc
        
    except Exception as e:
        print(f"Hata: Kontrast ayarlama işlemi uygulanırken bir hata oluştu: {str(e)}")
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

def goruntuleri_karsilastir(img1, img2, baslik1="Orijinal", baslik2="İşlenmiş", bekle=True):
    """
    İki görüntüyü yan yana gösterir.
    
    Args:
        img1 (numpy.ndarray): İlk görüntü
        img2 (numpy.ndarray): İkinci görüntü
        baslik1 (str, optional): İlk görüntünün başlığı. Varsayılan "Orijinal".
        baslik2 (str, optional): İkinci görüntünün başlığı. Varsayılan "İşlenmiş".
        bekle (bool, optional): True ise bir tuşa basılana kadar bekler. Varsayılan True.
    """
    if img1 is None or img2 is None:
        print("Hata: Karşılaştırılacak görüntülerden biri eksik!")
        return
        
    try:
        cv2.imshow(baslik1, img1)
        cv2.imshow(baslik2, img2)
        
        if bekle:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.waitKey(1)
            
    except Exception as e:
        print(f"Hata: Görüntüler karşılaştırılırken bir hata oluştu: {str(e)}")