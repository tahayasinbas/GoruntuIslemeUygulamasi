import cv2
import numpy as np
import matplotlib.pyplot as plt

def ortalama_filtre(img, kernel_size=5):
    """
    Görüntüye ortalama (mean) filtresi uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre pencere boyutu (tek sayı olmalı). Varsayılan 5.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Tek sayı kontrolü
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"Uyarı: Çekirdek boyutu çift sayı olamaz. {kernel_size} olarak ayarlandı.")
        
        # OpenCV'nin built-in blur fonksiyonu ile ortalama filtresi
        filtered_img = cv2.blur(img, (kernel_size, kernel_size))
        return filtered_img
        
    except Exception as e:
        print(f"Hata: Ortalama filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def ortalama_filtre_manuel(img, kernel_size=3):
    """
    Görüntüye manuel olarak ortalama (mean) filtresi uygular.
    OpenCV kullanmadan, manuel olarak filtreleme yapar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre pencere boyutu (tek sayı olmalı). Varsayılan 3.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Tek sayı kontrolü
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"Uyarı: Çekirdek boyutu çift sayı olamaz. {kernel_size} olarak ayarlandı.")
        
        # Görüntü boyutlarını al
        if len(img.shape) == 2:  # Gri tonlamalı görüntü
            h, w = img.shape
            channels = 1
        else:  # Renkli görüntü
            h, w, channels = img.shape
        
        # Çıktı görüntüsü oluştur
        output = np.zeros_like(img)
        
        # Filtre yarıçapı (yarı pencere boyutu)
        radius = kernel_size // 2
        
        # Filtreleme işlemi
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                # Pencere içindeki pikselleri topla
                if channels == 1:  # Gri tonlamalı
                    window = img[i - radius:i + radius + 1, j - radius:j + radius + 1]
                    output[i, j] = np.mean(window)
                else:  # Renkli
                    for c in range(channels):
                        window = img[i - radius:i + radius + 1, j - radius:j + radius + 1, c]
                        output[i, j, c] = np.mean(window)
        
        return output
        
    except Exception as e:
        print(f"Hata: Manuel ortalama filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def medyan_filtre(img, kernel_size=5):
    """
    Görüntüye medyan filtresi uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre pencere boyutu (tek sayı olmalı). Varsayılan 5.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Tek sayı kontrolü
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"Uyarı: Çekirdek boyutu çift sayı olamaz. {kernel_size} olarak ayarlandı.")
        
        # OpenCV'nin built-in medianBlur fonksiyonu ile medyan filtresi
        filtered_img = cv2.medianBlur(img, kernel_size)
        return filtered_img
        
    except Exception as e:
        print(f"Hata: Medyan filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def medyan_filtre_manuel(img, kernel_size=3):
    """
    Görüntüye manuel olarak medyan filtresi uygular.
    OpenCV kullanmadan, manuel olarak filtreleme yapar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre pencere boyutu (tek sayı olmalı). Varsayılan 3.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Tek sayı kontrolü
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"Uyarı: Çekirdek boyutu çift sayı olamaz. {kernel_size} olarak ayarlandı.")
        
        # Görüntü boyutlarını al
        if len(img.shape) == 2:  # Gri tonlamalı görüntü
            h, w = img.shape
            channels = 1
        else:  # Renkli görüntü
            h, w, channels = img.shape
        
        # Çıktı görüntüsü oluştur
        output = np.zeros_like(img)
        
        # Filtre yarıçapı (yarı pencere boyutu)
        radius = kernel_size // 2
        
        # Filtreleme işlemi
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                # Pencere içindeki pikselleri topla
                if channels == 1:  # Gri tonlamalı
                    window = img[i - radius:i + radius + 1, j - radius:j + radius + 1]
                    output[i, j] = np.median(window)
                else:  # Renkli
                    for c in range(channels):
                        window = img[i - radius:i + radius + 1, j - radius:j + radius + 1, c]
                        output[i, j, c] = np.median(window)
        
        return output
        
    except Exception as e:
        print(f"Hata: Manuel medyan filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def gauss_filtre(img, kernel_size=5, sigma=1.0):
    """
    Görüntüye Gauss filtresi uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre pencere boyutu (tek sayı olmalı). Varsayılan 5.
        sigma (float, optional): Gauss standardı sapma parametresi. Varsayılan 1.0.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Tek sayı kontrolü
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"Uyarı: Çekirdek boyutu çift sayı olamaz. {kernel_size} olarak ayarlandı.")
        
        # OpenCV'nin built-in GaussianBlur fonksiyonu ile Gauss filtresi
        filtered_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        return filtered_img
        
    except Exception as e:
        print(f"Hata: Gauss filtresi uygulanırken bir hata oluştu: {str(e)}")
        return None

def gauss_filtre_manuel(img, kernel_size=5, sigma=1.0):
    """
    Görüntüye manuel olarak oluşturulan Gauss filtresi uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre pencere boyutu (tek sayı olmalı). Varsayılan 5.
        sigma (float, optional): Gauss standardı sapma parametresi. Varsayılan 1.0.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Tek sayı kontrolü
        if kernel_size % 2 == 0:
            kernel_size += 1
            print(f"Uyarı: Çekirdek boyutu çift sayı olamaz. {kernel_size} olarak ayarlandı.")
        
        # Gauss çekirdeği oluştur
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Gauss fonksiyonu ile çekirdek değerlerini hesapla
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Çekirdeği normalize et (toplamı 1 olacak şekilde)
        kernel = kernel / np.sum(kernel)
        
        # Konvolüsyon işlemi (filtreleme)
        if len(img.shape) == 2:  # Gri tonlamalı görüntü
            filtered_img = cv2.filter2D(img, -1, kernel)
        else:  # Renkli görüntü
            filtered_img = np.zeros_like(img)
            for c in range(img.shape[2]):
                filtered_img[:,:,c] = cv2.filter2D(img[:,:,c], -1, kernel)
        
        return filtered_img
        
    except Exception as e:
        print(f"Hata: Manuel Gauss filtresi uygulanırken bir hata oluştu: {str(e)}")
        return None

def konservatif_filtre(img):
    """
    Görüntüye konservatif (muhafazakar) filtreleme uygular.
    Bu filtre, bir pikselin değerini sadece komşu piksellerin
    min ve max değerleri arasında değiştirir.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Sonuç görüntüsü için kopya oluştur
        filtered_img = img.copy()
        
        # Görüntü boyutları
        if len(img.shape) == 2:  # Gri tonlamalı görüntü
            rows, cols = img.shape
            channels = 1
        else:  # Renkli görüntü
            rows, cols, channels = img.shape
        
        # Filtreleme işlemi - sınır pikseller hariç
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if channels == 1:  # Gri tonlamalı
                    # 3x3 penceresi
                    neighbors = img[i-1:i+2, j-1:j+2]
                    min_val = np.min(neighbors)
                    max_val = np.max(neighbors)
                    
                    # Piksel değerini min ve max arasında tut
                    if img[i,j] < min_val:
                        filtered_img[i,j] = min_val
                    elif img[i,j] > max_val:
                        filtered_img[i,j] = max_val
                
                else:  # Renkli görüntü
                    for c in range(channels):
                        # 3x3 penceresi (her kanal için)
                        neighbors = img[i-1:i+2, j-1:j+2, c]
                        min_val = np.min(neighbors)
                        max_val = np.max(neighbors)
                        
                        # Piksel değerini min ve max arasında tut
                        if img[i,j,c] < min_val:
                            filtered_img[i,j,c] = min_val
                        elif img[i,j,c] > max_val:
                            filtered_img[i,j,c] = max_val
        
        return filtered_img
        
    except Exception as e:
        print(f"Hata: Konservatif filtreleme uygulanırken bir hata oluştu: {str(e)}")
        return None

def crimmins_speckle_filtre(img, iterations=1):
    """
    Görüntüye Crimmins Speckle Removal (beneklenme giderme) filtresi uygular.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        iterations (int, optional): Filtreleme iterasyon sayısı. Varsayılan 1.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Renkli görüntüyü gri tonlamaya çevir
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Sonuç görüntüsü
        result = gray.copy()
        
        # Belirlenen iterasyon sayısı kadar tekrarla
        for _ in range(iterations):
            temp = result.copy()
            rows, cols = temp.shape
            
            # Her piksel için (sınırları hariç tut)
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = temp[i, j]
                    # 4-komşuluk (üst, alt, sol, sağ)
                    neighbors = [temp[i-1, j], temp[i+1, j], temp[i, j-1], temp[i, j+1]]
                    
                    # Komşuların ortalaması
                    avg_neighbors = sum(neighbors) / len(neighbors)
                    
                    # Eğer merkez piksel belirli bir eşikten daha parlak/karanlıksa düzelt
                    if center > avg_neighbors + 20:  # Parlak beneklenme (salt)
                        result[i, j] = avg_neighbors
                    elif center < avg_neighbors - 20:  # Karanlık beneklenme (pepper)
                        result[i, j] = avg_neighbors
        
        return result
        
    except Exception as e:
        print(f"Hata: Crimmins Speckle filtresi uygulanırken bir hata oluştu: {str(e)}")
        return None

def fft_goruntu(img):
    """
    Görüntünün Fourier dönüşümünü hesaplar ve merkezlenmiş büyüklük spektrumunu döndürür.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        
    Returns:
        tuple: (fft_shifted, magnitude_spectrum) - kaydırılmış FFT ve büyüklük spektrumu
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Fourier dönüşümü uygula
        f_transform = np.fft.fft2(gray)
        # Düşük frekansları merkeze kaydır
        f_transform_shifted = np.fft.fftshift(f_transform)
        # Genlik spektrumunu logaritmik ölçekte hesapla
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
        
        return f_transform_shifted, magnitude_spectrum
        
    except Exception as e:
        print(f"Hata: Fourier dönüşümü hesaplanırken bir hata oluştu: {str(e)}")
        return None, None

def goruntu_fft_goster(img, baslik="Fourier Dönüşümü"):
    """
    Görüntünün hem kendisini hem de Fourier dönüşümünü gösterir.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        baslik (str, optional): Görüntü başlığı. Varsayılan "Fourier Dönüşümü".
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Fourier dönüşümünü al
        _, magnitude_spectrum = fft_goruntu(gray)
        
        # Görüntüleri göster
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title(f"Orijinal {baslik}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title(f"FFT Magnitude Spectrum {baslik}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Hata: Fourier dönüşümü gösterilirken bir hata oluştu: {str(e)}")

def alcak_geciren_filtre(img, kesme_frekansi=30):
    """
    Görüntüye Fourier dönüşümü kullanarak alçak geçiren filtre uygular (LPF).
    Düşük frekansları (merkez bölge) korur, yüksek frekansları (detayları) azaltır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        kesme_frekansi (int, optional): Kesme frekansı (filtre yarıçapı). Varsayılan 30.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Alçak geçiren filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.uint8)
        center = (cols // 2, rows // 2)
        cv2.circle(mask, center, kesme_frekansi, 1, -1)
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Alçak geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def yuksek_geciren_filtre(img, kesme_frekansi=30):
    """
    Görüntüye Fourier dönüşümü kullanarak yüksek geçiren filtre uygular (HPF).
    Yüksek frekansları (detayları) korur, düşük frekansları (merkez bölge) azaltır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        kesme_frekansi (int, optional): Kesme frekansı (filtre yarıçapı). Varsayılan 30.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Yüksek geçiren filtre maskesi oluştur (alçak geçiren maskesinin tersi)
        mask = np.ones((rows, cols), np.uint8)
        center = (cols // 2, rows // 2)
        cv2.circle(mask, center, kesme_frekansi, 0, -1)
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Yüksek geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def butterworth_alcak_geciren_filtre(img, kesme_frekansi=30, derece=2):
    """
    Görüntüye Butterworth alçak geçiren filtre uygular.
    Daha yumuşak geçişli bir filtre, ideal filtreden farklı olarak halkalanma etkisini azaltır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        kesme_frekansi (int, optional): Kesme frekansı. Varsayılan 30.
        derece (int, optional): Filtre derecesi. Yüksek değerler geçişi keskinleştirir. Varsayılan 2.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2  # Merkez nokta
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Butterworth alçak geçiren filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                # Butterworth filtre formulü
                mask[i, j] = 1 / (1 + (d / kesme_frekansi) ** (2 * derece))
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Butterworth alçak geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def butterworth_yuksek_geciren_filtre(img, kesme_frekansi=30, derece=2):
    """
    Görüntüye Butterworth yüksek geçiren filtre uygular.
    Daha yumuşak geçişli bir yüksek geçiren filtre.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        kesme_frekansi (int, optional): Kesme frekansı. Varsayılan 30.
        derece (int, optional): Filtre derecesi. Yüksek değerler geçişi keskinleştirir. Varsayılan 2.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2  # Merkez nokta
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Butterworth yüksek geçiren filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                # Butterworth filtre formulü (yüksek geçiren)
                if d == 0:  # Sıfıra bölme hatasını engellemek için
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1 / (1 + (kesme_frekansi / d) ** (2 * derece))
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Butterworth yüksek geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def band_geciren_filtre(img, ic_yaricap=30, dis_yaricap=50):
    """
    Görüntüye band geçiren filtre uygular (BPF).
    Belirli frekans aralığındaki (bantlar) frekansları korur, diğerlerini filtreler.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        ic_yaricap (int, optional): İç yarıçap (minimum frekans). Varsayılan 30.
        dis_yaricap (int, optional): Dış yarıçap (maksimum frekans). Varsayılan 50.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Band geçiren filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.uint8)
        center = (cols // 2, rows // 2)
        
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                # Belirli frekans bandını geçir
                if ic_yaricap <= d <= dis_yaricap:
                    mask[i, j] = 1
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Band geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def band_durduran_filtre(img, ic_yaricap=30, dis_yaricap=50):
    """
    Görüntüye band durduran filtre uygular (BSF).
    Belirli frekans aralığındaki (bantlar) frekansları durdurur, diğerlerini geçirir.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        ic_yaricap (int, optional): İç yarıçap (minimum frekans). Varsayılan 30.
        dis_yaricap (int, optional): Dış yarıçap (maksimum frekans). Varsayılan 50.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Band durduran filtre maskesi oluştur (band geçiren maskesinin tersi)
        mask = np.ones((rows, cols), np.uint8)
        center = (cols // 2, rows // 2)
        
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                # Belirli frekans bandını durdur
                if ic_yaricap <= d <= dis_yaricap:
                    mask[i, j] = 0
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Band durduran filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def butterworth_band_geciren_filtre(img, ic_yaricap=30, dis_yaricap=50, derece=2):
    """
    Görüntüye Butterworth band geçiren filtre uygular.
    Yuvarma fonksiyonu sayesinde keskin frekans geçişleri yerine yuvarlanmış geçişler sağlar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        ic_yaricap (int, optional): İç kesme frekansı. Varsayılan 30.
        dis_yaricap (int, optional): Dış kesme frekansı. Varsayılan 50.
        derece (int, optional): Filtre derecesi. Yüksek değerler geçişi keskinleştirir. Varsayılan 2.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2  # Merkez nokta
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Butterworth band geçiren filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                # Band geçiren Butterworth filtre formulü:
                # H(u,v) = 1 - 1 / (1 + (d/D0)^2n) formulünde d merkeze uzaklık, D0 kesme frekansı ve n derece
                if d == 0:  # Sıfıra bölme hatasını önle
                    mask[i, j] = 0
                else:
                    # İç frekansın altındakileri azalt
                    h1 = 1 / (1 + (ic_yaricap / d) ** (2 * derece))
                    # Dış frekansın üstündekileri azalt
                    h2 = 1 / (1 + (d / dis_yaricap) ** (2 * derece))
                    # İki filtreyi birleştir
                    mask[i, j] = h1 * h2
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Butterworth band geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def butterworth_band_durduran_filtre(img, ic_yaricap=30, dis_yaricap=50, derece=2):
    """
    Görüntüye Butterworth band durduran filtre uygular.
    Belirli bir bandı yuvarlanmış geçişlerle durdurur.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        ic_yaricap (int, optional): İç kesme frekansı. Varsayılan 30.
        dis_yaricap (int, optional): Dış kesme frekansı. Varsayılan 50.
        derece (int, optional): Filtre derecesi. Yüksek değerler geçişi keskinleştirir. Varsayılan 2.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2  # Merkez nokta
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Butterworth band durduran filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                
                # Ortadaki (dis_yaricap, ic_yaricap) bandı için dönüşüm
                if d == 0:  # Sıfıra bölme hatasını önle
                    mask[i, j] = 1
                else:
                    # Band durduran filtre formülü
                    # 1 - [d^2*W / (d^2 - d0^2)^2]^n 
                    # Burada d0 merkez frekans, W bant genişliği
                    d0_kare = ic_yaricap * dis_yaricap
                    w_kare = dis_yaricap - ic_yaricap
                    
                    # Band durduran Butterworth formülü
                    # 1 / (1 + [(d*W) / (d^2 - d0^2)]^2n)
                    payda = (d**2 - d0_kare) ** 2
                    if payda == 0:  # Bölme hatasını önle
                        mask[i, j] = 0
                    else:
                        pay = (d * w_kare) ** 2
                        mask[i, j] = 1 / (1 + (pay / payda) ** derece)
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Butterworth band durduran filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def gaussian_alcak_geciren_filtre(img, kesme_frekansi=30):
    """
    Görüntüye Gaussian alçak geçiren filtre uygular.
    Gauss fonksiyonu ile yuvarlanmış geçişli bir filtre oluşturur.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        kesme_frekansi (int, optional): Kesme frekansı (standart sapma). Varsayılan 30.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Gaussian alçak geçiren filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                # Gaussian formülü H(u,v) = e^(-D^2/(2*D0^2))
                mask[i, j] = np.exp(-(d**2) / (2 * (kesme_frekansi**2)))
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Gaussian alçak geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def gaussian_yuksek_geciren_filtre(img, kesme_frekansi=30):
    """
    Görüntüye Gaussian yüksek geçiren filtre uygular.
    Alçak geçiren Gaussian filtrenin tam tersi olarak çalışır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü (gri tonlamalı olmalı)
        kesme_frekansi (int, optional): Kesme frekansı (standart sapma). Varsayılan 30.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Fourier dönüşümünü al
        f_transform_shifted, _ = fft_goruntu(gray)
        
        # Gaussian yüksek geçiren filtre maskesi oluştur (1 - alçak geçiren)
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                # Gaussian formülü H(u,v) = 1 - e^(-D^2/(2*D0^2))
                mask[i, j] = 1 - np.exp(-(d**2) / (2 * (kesme_frekansi**2)))
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Görüntüyü 0-255 aralığına normalize et
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        print(f"Hata: Gaussian yüksek geçiren filtre uygulanırken bir hata oluştu: {str(e)}")
        return None


def homomorfik_filtre(img, d0=30, h_l=0.5, h_h=2.0, c=1):
    """
    Görüntüye homomorfik filtre uygular. Aydınlatma varyasyonlarını azaltırken, 
    görüntünün kontrasını artırır. Özellikle aydınlatma farklılıkları olan görüntülerde kullanışlıdır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        d0 (int, optional): Kesme frekansı. Varsayılan 30.
        h_l (float, optional): Düşük frekans kazanç faktörü. Varsayılan 0.5.
        h_h (float, optional): Yüksek frekans kazanç faktörü. Varsayılan 2.0.
        c (int, optional): Keskinlik kontrolü. Varsayılan 1.
        
    Returns:
        numpy.ndarray: Filtrelenmiş görüntü
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Logaritmik dönüşüm (görüntü piksel değerleri > 0 olmalı)
        # Logaritma, çarpma işlemlerini toplama işlemlerine dönüştürür
        # I(x,y) = i(x,y) * r(x,y) => log(I) = log(i) + log(r)
        # Burada i aydınlatma, r yansıtmadır
        log_image = np.log1p(np.float32(gray)) # log1p: log(1+x)
        
        # Fourier dönüşümü
        f_transform = np.fft.fft2(log_image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # Görüntü boyutları
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Homomorfik filtre maskesi oluştur
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                # Merkeze olan uzaklık
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                # Homomorfik filtre formülü:
                # H(u,v) = (h_h - h_l) * (1 - exp(-c*(D^2/d0^2))) + h_l
                mask[i, j] = (h_h - h_l) * (1 - np.exp(-c * (d**2 / d0**2))) + h_l
        
        # Filtreyi uygula
        filtered_fft = f_transform_shifted * mask
        
        # Ters Fourier dönüşümü uygula
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
        
        # Üssel dönüşüm (logaritma işleminin tersi)
        # expm1: exp(x)-1
        final_image = np.expm1(filtered_image) 
        
        # Görüntüyü 0-255 aralığına normalize et
        final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return final_image
        
    except Exception as e:
        print(f"Hata: Homomorfik filtre uygulanırken bir hata oluştu: {str(e)}")
        return None

def iki_goruntu_karsilastir(img1, img2, baslik1="Orijinal", baslik2="Filtrelenmiş", cmap=None):
    """
    İki görüntüyü yan yana karşılaştır.
    
    Args:
        img1 (numpy.ndarray): İlk görüntü
        img2 (numpy.ndarray): İkinci görüntü
        baslik1 (str, optional): İlk görüntünün başlığı. Varsayılan "Orijinal".
        baslik2 (str, optional): İkinci görüntünün başlığı. Varsayılan "Filtrelenmiş".
        cmap (str, optional): Renk haritası ('gray' gibi). Renkli görüntüler için None kullanın.
    """
    if img1 is None or img2 is None:
        print("Hata: Karşılaştırılacak görüntülerden biri eksik!")
        return
    
    try:
        plt.figure(figsize=(10, 5))
        
        # BGR to RGB dönüşümü (gerekirse)
        if len(img1.shape) == 3 and cmap is None:
            img1_show = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_show = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            img1_show = img1
            img2_show = img2
        
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
