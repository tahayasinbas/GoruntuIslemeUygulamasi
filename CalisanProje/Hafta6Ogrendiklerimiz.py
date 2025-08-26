import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Kenar Bulma Algoritmaları - 6. Hafta

Bu dosya çeşitli kenar bulma algoritmaları içerir:
1. Sobel - Gradyan tabanlı kenar algılama
2. Prewitt - Gradyan tabanlı kenar algılama
3. Roberts Cross - Gradyan tabanlı kenar algılama (daha küçük çekirdek)
4. Compass - Yön tabanlı kenar algılama
5. Canny - Çok aşamalı kenar algılama algoritması
'''

def sobel_kenar_bulma(img, ksize=3, normalize=True):
    """
    Sobel kenar bulma algoritması. X ve Y yönündeki gradyanları hesaplar ve
    kenar büyüklüğünü döndürür.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        ksize (int, optional): Filtre çekirdek boyutu. Varsayılan 3.
        normalize (bool, optional): Sonuçları 0-255 aralığına normalize etmek için. Varsayılan True.
        
    Returns:
        tuple: (sobel_x, sobel_y, sobel_magnitude) - X yönü, Y yönü ve toplam büyüklük
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None, None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # X ve Y yönündeki gradyanları hesapla
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Kenar büyüklüğünü hesapla (Gradient Magnitude)
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        
        # Normalize et (isteğe bağlı)
        if normalize:
            sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sobel_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return sobel_x, sobel_y, sobel_magnitude
        
    except Exception as e:
        print(f"Hata: Sobel kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None, None, None


def prewitt_kenar_bulma(img, normalize=True):
    """
    Prewitt kenar bulma algoritması. X ve Y yönündeki gradyanları hesaplar ve
    kenar büyüklüğünü döndürür.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        normalize (bool, optional): Sonuçları 0-255 aralığına normalize etmek için. Varsayılan True.
        
    Returns:
        tuple: (prewitt_x, prewitt_y, prewitt_magnitude) - X yönü, Y yönü ve toplam büyüklük
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None, None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Prewitt filtre çekirdeklerini tanımla
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # X ve Y yönündeki gradyanları hesapla
        prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        
        # Kenar büyüklüğünü hesapla
        prewitt_magnitude = cv2.magnitude(prewitt_x, prewitt_y)
        
        # Normalize et (isteğe bağlı)
        if normalize:
            prewitt_x = cv2.normalize(prewitt_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            prewitt_y = cv2.normalize(prewitt_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            prewitt_magnitude = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return prewitt_x, prewitt_y, prewitt_magnitude
        
    except Exception as e:
        print(f"Hata: Prewitt kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None, None, None


def roberts_cross_kenar_bulma(img, normalize=True):
    """
    Roberts Cross kenar bulma algoritması. 2x2 çekirdekler kullanarak çapraz gradyanları hesaplar.
    Roberts, daha küçük çekirdek boyutu nedeniyle güürültüye daha duyarlıdır ancak kenarları daha keskin bulabilir.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        normalize (bool, optional): Sonuçları 0-255 aralığına normalize etmek için. Varsayılan True.
        
    Returns:
        tuple: (roberts_x, roberts_y, roberts_magnitude) - X yönü, Y yönü ve toplam büyüklük
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None, None
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Roberts Cross filtre çekirdeklerini tanımla
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        # X ve Y yönündeki gradyanları hesapla
        roberts_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        roberts_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        
        # Kenar büyüklüğünü hesapla
        roberts_magnitude = cv2.magnitude(roberts_x, roberts_y)
        
        # Normalize et (isteğe bağlı)
        if normalize:
            roberts_x = cv2.normalize(roberts_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            roberts_y = cv2.normalize(roberts_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            roberts_magnitude = cv2.normalize(roberts_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return roberts_x, roberts_y, roberts_magnitude
        
    except Exception as e:
        print(f"Hata: Roberts Cross kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None, None, None


def compass_kenar_bulma(img, normalize=True):
    """
    Compass (Pusula) kenar bulma algoritması. Farklı yönlerdeki gradyanları hesaplayarak en güçlü kenarları bulur.
    Doğu, batı, kuzey, güney, kuzeydoğu, kuzeybatı, güneydoğu ve güneybatı yönlerindeki kenarları tespit eder.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        normalize (bool, optional): Sonuçları 0-255 aralığına normalize etmek için. Varsayılan True.
        
    Returns:
        numpy.ndarray: Tüm yönlerin maksimum kenar değerleri
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
        
        # Compass (Pusula) filtre çekirdeklerini tanımla
        compass_kernels = [
            np.array([[-1,-1,-1], [1,1,1], [1,1,1]]),  # Doğu (E)
            np.array([[1,1,1], [1,1,1], [-1,-1,-1]]),  # Batı (W)
            np.array([[-1,1,1], [-1,1,1], [-1,1,1]]),  # Kuzey (N)
            np.array([[1,1,-1], [1,1,-1], [1,1,-1]]),  # Güney (S)
            np.array([[-1,-1,1], [-1,1,1], [1,1,1]]),  # Kuzeydoğu (NE)
            np.array([[1,-1,-1], [1,1,-1], [1,1,1]]),  # Kuzeybatı (NW)
            np.array([[1,1,1], [1,1,-1], [1,-1,-1]]),  # Güneydoğu (SE)
            np.array([[1,1,1], [-1,1,1], [-1,-1,1]])   # Güneybatı (SW)
        ]
        
        # Tüm yönleri hesapla ve maksimum değeri al
        compass_edges = np.zeros_like(gray, dtype=np.float32)
        
        for kernel in compass_kernels:
            # Filtreyi uygula
            edge = cv2.filter2D(gray, cv2.CV_32F, kernel)
            # Her piksel için maksimum değeri al
            compass_edges = np.maximum(compass_edges, edge)
        
        # Normalize et (isteğe bağlı)
        if normalize:
            compass_edges = cv2.normalize(compass_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return compass_edges
        
    except Exception as e:
        print(f"Hata: Compass kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None


def canny_kenar_bulma(img, alt_esik=50, ust_esik=150, aperture_size=3):
    """
    Canny kenar bulma algoritması. Çok aşamalı kenar tespit algoritması.
    1. Gürültü azaltma
    2. Gradyan hesaplama
    3. Non-maximum suppression
    4. İkili eşikleme ve hystheresis ile kenar takibi
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        alt_esik (int, optional): Alt eşik değeri. Varsayılan 50.
        ust_esik (int, optional): Üst eşik değeri. Varsayılan 150.
        aperture_size (int, optional): Sobel filtresi için aperture boyutu. Varsayılan 3.
        
    Returns:
        numpy.ndarray: Canny kenar haritası
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
        
        # Canny kenar algılama algoritmasını uygula
        canny_edges = cv2.Canny(gray, alt_esik, ust_esik, apertureSize=aperture_size)
        
        return canny_edges
        
    except Exception as e:
        print(f"Hata: Canny kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None


def goruntu_goster(img, title="Görüntü", cmap=None):
    """
    Görüntüyü göstermek için yardımcı fonksiyon.
    
    Args:
        img (numpy.ndarray): Gösterilecek görüntü
        title (str, optional): Görüntü başlığı. Varsayılan "Görüntü".
        cmap (str, optional): Renk haritası ('gray' gibi). Renkli görüntüler için None kullanın.
    """
    if img is None:
        print("Hata: Gösterilecek görüntü bulunamadı!")
        return
    
    try:
        # BGR to RGB dönüşümü (gerekirse)
        if len(img.shape) == 3 and cmap is None:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_show = img
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img_show, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Hata: Görüntü gösterilirken bir hata oluştu: {str(e)}")


def kenar_karsilastir(img, algoritma='hepsi'):
    """
    Farklı kenar bulma algoritmalarının sonuçlarını karşılaştırır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        algoritma (str, optional): 'sobel', 'prewitt', 'roberts', 'compass', 'canny' veya 'hepsi'. Varsayılan 'hepsi'.
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
        
        if algoritma.lower() == 'sobel' or algoritma.lower() == 'hepsi':
            # Sobel kenar bulma
            sobel_x, sobel_y, sobel_magnitude = sobel_kenar_bulma(gray)
            
            # Sobel sonuçlarını göster
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1), plt.imshow(sobel_x, cmap='gray'), plt.title("Sobel X")
            plt.subplot(1, 3, 2), plt.imshow(sobel_y, cmap='gray'), plt.title("Sobel Y")
            plt.subplot(1, 3, 3), plt.imshow(sobel_magnitude, cmap='gray'), plt.title("Sobel Toplam")
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'prewitt' or algoritma.lower() == 'hepsi':
            # Prewitt kenar bulma
            prewitt_x, prewitt_y, prewitt_magnitude = prewitt_kenar_bulma(gray)
            
            # Prewitt sonuçlarını göster
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1), plt.imshow(prewitt_x, cmap='gray'), plt.title("Prewitt X")
            plt.subplot(1, 3, 2), plt.imshow(prewitt_y, cmap='gray'), plt.title("Prewitt Y")
            plt.subplot(1, 3, 3), plt.imshow(prewitt_magnitude, cmap='gray'), plt.title("Prewitt Toplam")
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'roberts' or algoritma.lower() == 'hepsi':
            # Roberts Cross kenar bulma
            roberts_x, roberts_y, roberts_magnitude = roberts_cross_kenar_bulma(gray)
            
            # Roberts sonuçlarını göster
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1), plt.imshow(roberts_x, cmap='gray'), plt.title("Roberts X")
            plt.subplot(1, 3, 2), plt.imshow(roberts_y, cmap='gray'), plt.title("Roberts Y")
            plt.subplot(1, 3, 3), plt.imshow(roberts_magnitude, cmap='gray'), plt.title("Roberts Toplam")
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'compass' or algoritma.lower() == 'hepsi':
            # Compass kenar bulma
            compass_edges = compass_kenar_bulma(gray)
            
            # Compass sonuçlarını göster
            plt.figure(figsize=(8, 6))
            plt.imshow(compass_edges, cmap='gray')
            plt.title("Compass Kenar Algılama")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'canny' or algoritma.lower() == 'hepsi':
            # Canny kenar bulma
            canny_edges = canny_kenar_bulma(gray, 50, 150)
            
            # Canny sonuçlarını göster
            plt.figure(figsize=(8, 6))
            plt.imshow(canny_edges, cmap='gray')
            plt.title("Canny Kenar Algılama")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Tüm algoritmaları birlikte karşılaştır
        if algoritma.lower() == 'hepsi':
            _, _, sobel = sobel_kenar_bulma(gray)
            _, _, prewitt = prewitt_kenar_bulma(gray)
            _, _, roberts = roberts_cross_kenar_bulma(gray)
            compass = compass_kenar_bulma(gray)
            canny = canny_kenar_bulma(gray, 50, 150)
            
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1), plt.imshow(gray, cmap='gray'), plt.title("Orijinal Görüntü")
            plt.subplot(2, 3, 2), plt.imshow(sobel, cmap='gray'), plt.title("Sobel")
            plt.subplot(2, 3, 3), plt.imshow(prewitt, cmap='gray'), plt.title("Prewitt")
            plt.subplot(2, 3, 4), plt.imshow(roberts, cmap='gray'), plt.title("Roberts Cross")
            plt.subplot(2, 3, 5), plt.imshow(compass, cmap='gray'), plt.title("Compass")
            plt.subplot(2, 3, 6), plt.imshow(canny, cmap='gray'), plt.title("Canny")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Hata: Kenar bulma algoritmaları karşılaştırılırken bir hata oluştu: {str(e)}")
def laplace_kenar_bulma(img, ksize=3, normalize=True):
    """
    Laplace kenar bulma algoritması. İkinci türev tabanlı kenar tespiti yapar.
    Pozitif ve negatif kenarları tespit eder.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        ksize (int, optional): Filtre çekirdek boyutu. Varsayılan 3.
        normalize (bool, optional): Sonuçları 0-255 aralığına normalize etmek için. Varsayılan True.
        
    Returns:
        numpy.ndarray: Laplace kenar haritası
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
        
        # Laplace filtresini uygula
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        
        # Mutlak değeri al (pozitif ve negatif kenarlar için)
        laplacian_abs = np.abs(laplacian)
        
        # Normalize et (isteğe bağlı)
        if normalize:
            laplacian_abs = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return laplacian_abs
        
    except Exception as e:
        print(f"Hata: Laplace kenar bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None
def gabor_filtre(img, kernel_size=21, sigma=5, theta=np.pi/4, lambd=10, gamma=0.5, psi=0, normalize=True):
    """
    Gabor filtresi uygulayarak belirli yönelimde ve frekanstaki yapıları vurgular.
    Özellikle doku analizi ve nesne tanıma uygulamalarında kullanılır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kernel_size (int, optional): Filtre çekirdek boyutu. Varsayılan 21.
        sigma (float, optional): Gaussian standart sapması. Varsayılan 5.
        theta (float, optional): Yönelim açısı (radyan). Varsayılan pi/4.
        lambd (float, optional): Sinüs dalgasının dalga boyu. Varsayılan 10.
        gamma (float, optional): Uzamsal en-boy oranı. Varsayılan 0.5.
        psi (float, optional): Faz kayması. Varsayılan 0.
        normalize (bool, optional): Sonuçları 0-255 aralığına normalize etmek için. Varsayılan True.
        
    Returns:
        numpy.ndarray: Gabor filtresi uygulanmış görüntü
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
        
        # Gabor çekirdeğini oluştur
        gabor_kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), 
            sigma, 
            theta, 
            lambd, 
            gamma, 
            psi, 
            ktype=cv2.CV_32F
        )
        
        # Filtre uygula
        gabor_image = cv2.filter2D(gray, cv2.CV_64F, gabor_kernel)
        
        # Normalize et (isteğe bağlı)
        if normalize:
            gabor_image = cv2.normalize(gabor_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return gabor_image
        
    except Exception as e:
        print(f"Hata: Gabor filtresi uygulanırken bir hata oluştu: {str(e)}")
        return None
def hough_dogru_bulma(img, rho=1, theta=np.pi/180, esik=150, min_cizgi_uzunlugu=None, max_bosluk=None):
    """
    Hough dönüşümü kullanarak görüntüdeki doğruları tespit eder.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        rho (int, optional): Mesafe çözünürlüğü (piksel). Varsayılan 1.
        theta (float, optional): Açı çözünürlüğü (radyan). Varsayılan pi/180 (1 derece).
        esik (int, optional): Akümülatör eşik değeri. Varsayılan 150.
        min_cizgi_uzunlugu (float, optional): Minimum çizgi uzunluğu. None ise standart HoughLines kullanılır.
        max_bosluk (float, optional): İki çizgi parçası arasındaki maksimum boşluk. None ise standart HoughLines kullanılır.
        
    Returns:
        tuple: (orijinal görüntü, kenarlar, doğrular çizilmiş görüntü)
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None, None
    
    try:
        # Görüntüyü kopyala
        output = img.copy()
        
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            # Renkli bir görüntüye dönüştür (çizgileri renkli göstermek için)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Kenarları bul (Canny algoritması)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Probabilistic Hough Transform veya Standard Hough Transform kullan
        if min_cizgi_uzunlugu is not None and max_bosluk is not None:
            # Probabilistic Hough Transform
            lines = cv2.HoughLinesP(edges, rho, theta, esik, minLineLength=min_cizgi_uzunlugu, maxLineGap=max_bosluk)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Standard Hough Transform
            lines = cv2.HoughLines(edges, rho, theta, esik)
            
            if lines is not None:
                for line in lines:
                    rho_val, theta_val = line[0]
                    a = np.cos(theta_val)
                    b = np.sin(theta_val)
                    x0 = a * rho_val
                    y0 = b * rho_val
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return gray, edges, output
        
    except Exception as e:
        print(f"Hata: Hough doğru bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None, None, None


def hough_cember_bulma(img, dp=1.0, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=100):
    """
    Hough dönüşümü kullanarak görüntüdeki çemberleri tespit eder.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        dp (float, optional): Akümülatör çözünürlük oranı. Varsayılan 1.0.
        min_dist (int, optional): Algılanan çemberler arasındaki minimum mesafe. Varsayılan 30.
        param1 (int, optional): Canny kenar dedektörü için yüksek eşik. Varsayılan 50.
        param2 (int, optional): Algılama için akümülatör eşik değeri. Varsayılan 30.
        min_radius (int, optional): Minimum çember yarıçapı. Varsayılan 10.
        max_radius (int, optional): Maksimum çember yarıçapı. Varsayılan 100.
        
    Returns:
        tuple: (orijinal görüntü, çemberler çizilmiş görüntü)
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None
    
    try:
        # Görüntüyü kopyala
        output = img.copy()
        
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            # Renkli bir görüntüye dönüştür (çemberleri renkli göstermek için)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough çember dönüşümünü uygula
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                                  param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
        
        # Çemberleri çiz
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Dış çemberi çiz
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Merkez noktasını çiz
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        return gray, output
        
    except Exception as e:
        print(f"Hata: Hough çember bulma algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None, None
def kmeans_segmentation(img, k=3, attempts=10):
    """
    K-Means kümeleme algoritması kullanarak görüntü segmentasyonu yapar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        k (int, optional): Küme (segment) sayısı. Varsayılan 3.
        attempts (int, optional): Algoritmanın farklı başlangıç noktalarıyla kaç kez çalıştırılacağı. Varsayılan 10.
        
    Returns:
        tuple: (orijinal görüntü, segmente edilmiş görüntü, etiketler)
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return None, None, None
    
    try:
        # OpenCV BGR formatındaysa RGB'ye dönüştür
        if len(img.shape) == 3:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Gri tonlamalı görüntüyü 3 kanallı yap
            image_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Görüntüyü 2D vektör haline getir (satır x sütun, 3 kanal)
        pixel_values = image_rgb.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # K-Means algoritması için sonlandırma kriterleri
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # K-Means algoritmasını uygula
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        
        # Vektörleri tekrar RGB değerlerine dönüştür
        centers = np.uint8(centers)
        
        # Her pikseli küme merkeziyle değiştir
        segmented_data = centers[labels.flatten()]
        
        # Tekrar görüntü formatına getir
        segmented_image = segmented_data.reshape(image_rgb.shape)
        
        # Etiketleri görselleştirmek için tekrar şekillendir (2D görüntü formatı)
        label_image = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])
        
        return image_rgb, segmented_image, label_image
        
    except Exception as e:
        print(f"Hata: K-Means segmentasyon algoritması uygulanırken bir hata oluştu: {str(e)}")
        return None, None, None
def gelismis_kenar_karsilastir(img, algoritma='hepsi'):
    """
    Gelişmiş kenar bulma ve segmentasyon algoritmalarının sonuçlarını karşılaştırır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        algoritma (str, optional): 'laplace', 'gabor', 'hough_dogru', 'hough_cember', 'kmeans' veya 'hepsi'. Varsayılan 'hepsi'.
    """
    if img is None:
        print("Hata: İşlenecek görüntü bulunamadı!")
        return
    
    try:
        # Görüntü gri tonlamalı değilse, gri tonlamaya çevir
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            color_img = img.copy()
        else:
            gray = img.copy()
            color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        if algoritma.lower() == 'laplace' or algoritma.lower() == 'hepsi':
            # Laplace kenar bulma
            laplacian = laplace_kenar_bulma(gray)
            
            # Laplace sonuçlarını göster
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(gray, cmap='gray'), plt.title("Orijinal Görüntü")
            plt.subplot(1, 2, 2), plt.imshow(laplacian, cmap='gray'), plt.title("Laplace Kenar Algılama")
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'gabor' or algoritma.lower() == 'hepsi':
            # Farklı yönelimlerde Gabor filtreleri uygula
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1), plt.imshow(gray, cmap='gray'), plt.title("Orijinal Görüntü")
            
            for i, angle in enumerate(angles):
                gabor_result = gabor_filtre(gray, theta=angle)
                plt.subplot(2, 3, i+2), plt.imshow(gabor_result, cmap='gray')
                plt.title(f"Gabor (θ={angle:.2f})")
            
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'hough_dogru' or algoritma.lower() == 'hepsi':
            # Hough doğru dönüşümü
            _, edges, hough_lines = hough_dogru_bulma(color_img)
            
            # Hough doğru sonuçlarını göster
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title("Orijinal Görüntü")
            plt.subplot(1, 3, 2), plt.imshow(edges, cmap='gray'), plt.title("Kenarlar")
            plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(hough_lines, cv2.COLOR_BGR2RGB)), plt.title("Hough Doğruları")
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'hough_cember' or algoritma.lower() == 'hepsi':
            # Hough çember dönüşümü
            _, hough_circles = hough_cember_bulma(color_img)
            
            # Hough çember sonuçlarını göster
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title("Orijinal Görüntü")
            plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(hough_circles, cv2.COLOR_BGR2RGB)), plt.title("Hough Çemberleri")
            plt.tight_layout()
            plt.show()
        
        if algoritma.lower() == 'kmeans' or algoritma.lower() == 'hepsi':
            # K-Means segmentasyonu
            _, segmented, _ = kmeans_segmentation(color_img, k=4)
            
            # K-Means sonuçlarını göster
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title("Orijinal Görüntü")
            plt.subplot(1, 2, 2), plt.imshow(segmented), plt.title("K-Means Segmentasyonu (k=4)")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Hata: Gelişmiş algoritmalar karşılaştırılırken bir hata oluştu: {str(e)}")