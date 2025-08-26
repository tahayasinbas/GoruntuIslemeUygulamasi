import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Morfolojik İşlemler - 7. Hafta

Bu dosya çeşitli morfolojik işlem fonksiyonları içerir:
1. Erode (Aşındırma) - Nesneleri küçültme, ince ayrıntıları kaldırma
2. Dilate (Genişletme) - Nesneleri büyütme, boşlukları doldurma
3. Morfolojik işlem kombinasyonları (Opening, Closing, vb.)
'''

def erode_islem(img, kernel_size=3, kernel_shape="kare", iterations=1):
    """
    Aşındırma (Erosion) işlemi uygular. Bu işlem, görüntüdeki nesneleri küçültür ve ince ayrıntıları kaldırır.
    Beyaz nesnelerin kenarlarını aşındırarak nesneleri küçültür ve küçük beyaz nesneleri tamamen kaldırır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Yapısal eleman boyutu. Varsayılan 3.
        kernel_shape (str, optional): Yapısal elemanın şekli. 'kare', 'disk', 'çapraz', 'elips' olabilir. Varsayılan 'kare'.
        iterations (int, optional): İşlemin tekrarlanma sayısı. Varsayılan 1.
        
    Returns:
        numpy.ndarray: Aşındırılmış görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntüyü işlem için uygun formata getir
        if len(img.shape) > 2:
            # Renkli görüntüyse, her kanal için ayrı işlem yap
            # Bunun için görüntüyü BGR kanallarına ayır ve her kanalı ayrı işle
            b, g, r = cv2.split(img)
            processed_image = img.copy()
        else:
            # Gri tonlamalı görüntü ise doğrudan işle
            processed_image = img.copy()
            b, g, r = None, None, None
        
        # Yapısal elemanı oluştur
        if kernel_shape.lower() == "kare":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape.lower() == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "çapraz" or kernel_shape.lower() == "capraz":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "elips":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Varsayılan: kare
        
        # Aşındırma işlemini uygula
        if b is not None and g is not None and r is not None:
            b_eroded = cv2.erode(b, kernel, iterations=iterations)
            g_eroded = cv2.erode(g, kernel, iterations=iterations)
            r_eroded = cv2.erode(r, kernel, iterations=iterations)
            
            # Kanalları birleştir
            processed_image = cv2.merge([b_eroded, g_eroded, r_eroded])
        else:
            processed_image = cv2.erode(processed_image, kernel, iterations=iterations)
        
        return processed_image
        
    except Exception as e:
        print(f"Hata: Aşındırma işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None


def dilate_islem(img, kernel_size=3, kernel_shape="kare", iterations=1):
    """
    Genişletme (Dilation) işlemi uygular. Bu işlem, görüntüdeki nesneleri büyütür ve boşlukları doldurur.
    Beyaz nesnelerin alanlarını artırır, küçük siyah boşlukları kapatır ve nesneleri birbirine bağlar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Yapısal eleman boyutu. Varsayılan 3.
        kernel_shape (str, optional): Yapısal elemanın şekli. 'kare', 'disk', 'çapraz', 'elips' olabilir. Varsayılan 'kare'.
        iterations (int, optional): İşlemin tekrarlanma sayısı. Varsayılan 1.
        
    Returns:
        numpy.ndarray: Genişletilmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntüyü işlem için uygun formata getir
        if len(img.shape) > 2:
            # Renkli görüntüyse, her kanal için ayrı işlem yap
            b, g, r = cv2.split(img)
            processed_image = img.copy()
        else:
            # Gri tonlamalı görüntü ise doğrudan işle
            processed_image = img.copy()
            b, g, r = None, None, None
        
        # Yapısal elemanı oluştur
        if kernel_shape.lower() == "kare":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape.lower() == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "çapraz" or kernel_shape.lower() == "capraz":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "elips":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Varsayılan: kare
        
        # Genişletme işlemini uygula
        if b is not None and g is not None and r is not None:
            b_dilated = cv2.dilate(b, kernel, iterations=iterations)
            g_dilated = cv2.dilate(g, kernel, iterations=iterations)
            r_dilated = cv2.dilate(r, kernel, iterations=iterations)
            
            # Kanalları birleştir
            processed_image = cv2.merge([b_dilated, g_dilated, r_dilated])
        else:
            processed_image = cv2.dilate(processed_image, kernel, iterations=iterations)
        
        return processed_image
        
    except Exception as e:
        print(f"Hata: Genişletme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None


def opening_islem(img, kernel_size=3, kernel_shape="kare"):
    """
    Açma (Opening) işlemi uygular. Bu işlem, önce aşındırma sonra genişletme işlemidir.
    Küçük nesneleri kaldırır ve veri kaybetmeden ana nesnelerin şeklini korur.
    Gürültü temizlemek, bağlantıları koparmak için kullanışlıdır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Yapısal eleman boyutu. Varsayılan 3.
        kernel_shape (str, optional): Yapısal elemanın şekli. 'kare', 'disk', 'çapraz', 'elips' olabilir. Varsayılan 'kare'.
        
    Returns:
        numpy.ndarray: Açma işlemi uygulanmış görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Yapısal elemanı oluştur
        if kernel_shape.lower() == "kare":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape.lower() == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "çapraz" or kernel_shape.lower() == "capraz":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "elips":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Varsayılan: kare
        
        # Açma işlemi (erode, sonra dilate)
        processed_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        return processed_image
        
    except Exception as e:
        print(f"Hata: Açma işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None


def closing_islem(img, kernel_size=3, kernel_shape="kare"):
    """
    Kapama (Closing) işlemi uygular. Bu işlem, önce genişletme sonra aşındırma işlemidir.
    Nesne içindeki küçük boşlukları kapatır ve ana nesnelerin şeklini korur.
    Delikli nesneleri doldurmak, bağlantıları güçlendirmek için kullanışlıdır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Yapısal eleman boyutu. Varsayılan 3.
        kernel_shape (str, optional): Yapısal elemanın şekli. 'kare', 'disk', 'çapraz', 'elips' olabilir. Varsayılan 'kare'.
        
    Returns:
        numpy.ndarray: Kapama işlemi uygulanmış görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Yapısal elemanı oluştur
        if kernel_shape.lower() == "kare":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape.lower() == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "çapraz" or kernel_shape.lower() == "capraz":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "elips":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Varsayılan: kare
        
        # Kapama işlemi (dilate, sonra erode)
        processed_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return processed_image
        
    except Exception as e:
        print(f"Hata: Kapama işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None


def goruntu_karsilastir(img_original, img_processed, baslik1="Orijinal", baslik2="İşlenmiş", cmap=None):
    """
    Orijinal ve işlenmiş görüntüleri yan yana karşılaştırır.
    
    Args:
        img_original (numpy.ndarray): Orijinal görüntü
        img_processed (numpy.ndarray): İşlenmiş görüntü
        baslik1 (str, optional): İlk görüntünün başlığı. Varsayılan "Orijinal".
        baslik2 (str, optional): İkinci görüntünün başlığı. Varsayılan "İşlenmiş".
        cmap (str, optional): Renk haritası ('gray' gibi). Renkli görüntüler için None kullanın.
    """
    if img_original is None or img_processed is None:
        print("Hata: Gösterilecek görüntülerden biri eksik!")
        return
    
    try:
        # BGR to RGB dönüşümü (gerekirse)
        if len(img_original.shape) == 3 and cmap is None:
            img1_show = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            img2_show = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        else:
            img1_show = img_original
            img2_show = img_processed
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img1_show, cmap=cmap)
        plt.title(baslik1)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2_show, cmap=cmap)
        plt.title(baslik2)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Hata: Görüntüler karşılaştırılırken bir hata oluştu: {str(e)}")


def kernel_goster(kernel_size=3, kernel_shape="kare"):
    """
    Farklı şekillerdeki yapısal elemanları (kernel) gösterir.
    
    Args:
        kernel_size (int, optional): Yapısal eleman boyutu. Varsayılan 3.
        kernel_shape (str, optional): Yapısal elemanın şekli. 'kare', 'disk', 'çapraz', 'elips' olabilir. Varsayılan 'kare'.
    """
    try:
        # Yapısal elemanları oluştur
        kare_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        capraz_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elips_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Görüntüle
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(kare_kernel, cmap='gray')
        plt.title("Kare Kernel")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(disk_kernel, cmap='gray')
        plt.title("Disk Kernel")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(capraz_kernel, cmap='gray')
        plt.title("Çapraz Kernel")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(elips_kernel, cmap='gray')
        plt.title("Elips Kernel")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Hata: Kernel görüntülenirken bir hata oluştu: {str(e)}")


def morfolojik_islev_karsilastir(img, kernel_size=3, kernel_shape="kare"):
    """
    Farklı morfolojik işlemlerin sonuçlarını karşılaştırır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Yapısal eleman boyutu. Varsayılan 3.
        kernel_shape (str, optional): Yapısal elemanın şekli. 'kare', 'disk', 'çapraz', 'elips' olabilir. Varsayılan 'kare'.
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            gray = img.copy()
        
        # Tüm işlemleri uygula
        eroded_img = erode_islem(gray, kernel_size, kernel_shape)
        dilated_img = dilate_islem(gray, kernel_size, kernel_shape)
        opened_img = opening_islem(gray, kernel_size, kernel_shape)
        closed_img = closing_islem(gray, kernel_size, kernel_shape)
        
        # Sonuçları göster
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        if len(gray.shape) > 2:
            plt.imshow(gray)
        else:
            plt.imshow(gray, cmap='gray')
        plt.title("Orijinal Görüntü")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        if len(eroded_img.shape) > 2:
            plt.imshow(eroded_img)
        else:
            plt.imshow(eroded_img, cmap='gray')
        plt.title("Aşındırma (Erode)")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        if len(dilated_img.shape) > 2:
            plt.imshow(dilated_img)
        else:
            plt.imshow(dilated_img, cmap='gray')
        plt.title("Genişletme (Dilate)")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        if len(opened_img.shape) > 2:
            plt.imshow(opened_img)
        else:
            plt.imshow(opened_img, cmap='gray')
        plt.title("Açma (Opening)")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        if len(closed_img.shape) > 2:
            plt.imshow(closed_img)
        else:
            plt.imshow(closed_img, cmap='gray')
        plt.title("Kapama (Closing)")
        plt.axis('off')
        
        # Yapısal eleman bilgisi
        plt.subplot(2, 3, 6)
        if kernel_shape.lower() == "kare":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape.lower() == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "çapraz" or kernel_shape.lower() == "capraz":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape.lower() == "elips":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Varsayılan: kare
        
        plt.imshow(kernel * 255, cmap='gray')
        plt.title(f"Yapısal Eleman ({kernel_shape}, {kernel_size}x{kernel_size})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Hata: Morfolojik işlemler karşılaştırılırken bir hata oluştu: {str(e)}")