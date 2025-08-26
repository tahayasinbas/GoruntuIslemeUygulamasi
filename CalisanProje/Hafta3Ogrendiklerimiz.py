import cv2
import numpy as np
import matplotlib.pyplot as plt

def tasima(img, dx, dy):
    """
    Görüntüyü belirtilen miktarda x ve y eksenlerinde taşır.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        dx (int): X eksenindeki taşıma miktarı (pozitif: sağa, negatif: sola)
        dy (int): Y eksenindeki taşıma miktarı (pozitif: aşağı, negatif: yukarı)
        
    Returns:
        numpy.ndarray: Taşınmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Görüntü boyutları
        h, w = img.shape[:2]
        
        # Taşıma matrisi oluştur
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # Görüntüyü taşı
        moved_image = cv2.warpAffine(img, T, (w, h))
        
        return moved_image
        
    except Exception as e:
        print(f"Hata: Taşıma işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def aynalama_dikey(img):
    """
    Görüntüyü dikey eksende aynalar (soldan sağa çevirir).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        
    Returns:
        numpy.ndarray: Dikey eksende aynalanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # flip fonksiyonu: 1=yatay eksende aynalama (x ekseni), 0=dikey eksende aynalama (y ekseni)
        mirrored = cv2.flip(img, 1)  # Dikey eksen (x ekseni) etrafında aynalama
        return mirrored
        
    except Exception as e:
        print(f"Hata: Aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def aynalama_yatay(img):
    """
    Görüntüyü yatay eksende aynalar (yukarıdan aşağıya çevirir).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        
    Returns:
        numpy.ndarray: Yatay eksende aynalanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # flip fonksiyonu: 0=yatay eksende aynalama (y ekseni)
        mirrored = cv2.flip(img, 0)  # Yatay eksen (y ekseni) etrafında aynalama
        return mirrored
        
    except Exception as e:
        print(f"Hata: Aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def aynalama_her_iki_eksen(img):
    """
    Görüntüyü hem yatay hem dikey eksende aynalar (180 derece döndürmeye eşdeğer).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        
    Returns:
        numpy.ndarray: Her iki eksende aynalanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # flip fonksiyonu: -1=her iki eksende aynalama
        mirrored = cv2.flip(img, -1)  # Her iki eksende aynalama
        return mirrored
        
    except Exception as e:
        print(f"Hata: Aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def aynalama_nokta(img, x0, y0):
    """
    Görüntüyü belirtilen (x0, y0) noktasına göre aynalar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        x0 (int): Aynalama noktasının x koordinatı
        y0 (int): Aynalama noktasının y koordinatı
        
    Returns:
        numpy.ndarray: Belirtilen noktaya göre aynalanmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        h, w = img.shape[:2]
        result = img.copy()
        
        # Aynalama için her piksel üzerinde döngü
        for y1 in range(h):
            for x1 in range(w):
                # Aynalama formülleri
                x2 = int(-x1 + 2 * x0)
                y2 = int(-y1 + 2 * y0)
                
                # Hesaplanan konumun görüntü sınırları içinde olup olmadığını kontrol et
                if 0 <= x2 < w and 0 <= y2 < h:
                    result[y2, x2] = img[y1, x1]
        
        return result
        
    except Exception as e:
        print(f"Hata: Aynalama işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def egme_x(img, sh_x):
    """
    Görüntüyü X ekseninde eğer (shearing).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        sh_x (float): X eksenindeki eğme faktörü (pozitif: sağa, negatif: sola)
        
    Returns:
        numpy.ndarray: X ekseninde eğilmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        h, w = img.shape[:2]
        
        # Shearing dönüşüm matrisi
        S = np.float32([[1, sh_x, 0], [0, 1, 0]])
        
        # Yeni genişlik hesaplama (shearing ile genişleme olabilir)
        new_w = w + int(abs(sh_x) * h)
        
        # Shearing işlemi
        sheared_image = cv2.warpAffine(img, S, (new_w, h))
        
        return sheared_image
        
    except Exception as e:
        print(f"Hata: X ekseninde eğme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def egme_y(img, sh_y):
    """
    Görüntüyü Y ekseninde eğer (shearing).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        sh_y (float): Y eksenindeki eğme faktörü (pozitif: aşağı, negatif: yukarı)
        
    Returns:
        numpy.ndarray: Y ekseninde eğilmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        h, w = img.shape[:2]
        
        # Shearing dönüşüm matrisi
        S = np.float32([[1, 0, 0], [sh_y, 1, 0]])
        
        # Yeni yükseklik hesaplama (shearing ile genişleme olabilir)
        new_h = h + int(abs(sh_y) * w)
        
        # Shearing işlemi
        sheared_image = cv2.warpAffine(img, S, (w, new_h))
        
        return sheared_image
        
    except Exception as e:
        print(f"Hata: Y ekseninde eğme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None
def olcekleme(img, scale_factor, interpolation=cv2.INTER_LINEAR):
    """
    Görüntüyü belirtilen ölçek faktörüyle büyütür veya küçültür.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        scale_factor (float): Ölçek faktörü. 1.0'dan büyük değerler büyütme, 
                             küçük değerler küçültme yapar
        interpolation (int, optional): Interpolasyon yöntemi. 
                                     Varsayılan: cv2.INTER_LINEAR.
        
    Returns:
        numpy.ndarray: Ölçeklenmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        h, w = img.shape[:2]
        
        # Yeni boyutları hesapla
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Görüntüyü yeniden boyutlandır
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        return resized
        
    except Exception as e:
        print(f"Hata: Ölçekleme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def olcekleme_manuel(img, scale_factor):
    """
    Görüntüyü manuel olarak (piksel değiştirme ile) küçültür. 
    Bu fonksiyon sadece küçültme için uygundur (scale_factor > 1).
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        scale_factor (int): Küçültme faktörü (2 = yarı boyut, 3 = üçte bir boyut)
        
    Returns:
        numpy.ndarray: Küçültülmüş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    if scale_factor < 1:
        print("Hata: Ölçek faktörü 1'den büyük olmalıdır!")
        return None
    
    try:
        scale_factor = int(scale_factor)
        h, w = img.shape[:2]
        
        new_h, new_w = h // scale_factor, w // scale_factor
        
        # Küçültülmüş görüntü için boş bir array oluştur
        if len(img.shape) == 3:  # Renkli görüntü
            result = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
        else:  # Gri tonlamalı görüntü
            result = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # Her pixel'i manuel olarak doldur
        for y in range(new_h):
            for x in range(new_w):
                result[y, x] = img[y * scale_factor, x * scale_factor]
                
        return result
        
    except Exception as e:
        print(f"Hata: Manuel ölçekleme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def dondurme(img, angle, center=None, scale=1.0, interpolation=cv2.INTER_LINEAR):
    """
    Görüntüyü belirtilen açıda döndürür.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        angle (float): Döndürme açısı (derece, pozitif = saat yönünün tersine)
        center (tuple, optional): Döndürme merkezi (x, y). None ise merkez nokta kullanılır
        scale (float, optional): Ölçekleme faktörü. Varsayılan 1.0.
        interpolation (int, optional): Interpolasyon yöntemi. 
                                     Varsayılan: cv2.INTER_LINEAR.
        
    Returns:
        numpy.ndarray: Döndürülmüş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        h, w = img.shape[:2]
        
        # Döndürme merkezi belirlenmemişse görüntü merkezini kullan
        if center is None:
            center = (w // 2, h // 2)
        
        # Döndürme matrisi oluştur
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Döndürme işlemini uygula
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=interpolation)
        
        return rotated
        
    except Exception as e:
        print(f"Hata: Döndürme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def kirpma(img, x, y, genislik, yukseklik):
    """
    Görüntüyü belirtilen koordinatlardan başlayarak, verilen genişlik ve yükseklikte kırpar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        x (int): Kırpma bölgesinin sol üst köşesinin x koordinatı
        y (int): Kırpma bölgesinin sol üst köşesinin y koordinatı
        genislik (int): Kırpılacak bölgenin genişliği
        yukseklik (int): Kırpılacak bölgenin yüksekliği
        
    Returns:
        numpy.ndarray: Kırpılmış görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Görüntü boyutlarını kontrol et
        h, w = img.shape[:2]
        
        # Parametreleri görüntü sınırlarına göre ayarla
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        genislik = min(genislik, w - x)
        yukseklik = min(yukseklik, h - y)
        
        # Kırpma işlemini uygula
        cropped = img[y:y+yukseklik, x:x+genislik]
        
        return cropped
        
    except Exception as e:
        print(f"Hata: Kırpma işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

class MouseSeciciKirpma:
    """
    Mouse ile dikdörtgen çizerek görüntü kırpma işlemi yapan sınıf.
    """
    def __init__(self, window_name="Mouse ile Kırpma"):
        self.window_name = window_name
        self.img = None
        self.img_copy = None
        self.roi_pts = []
        self.drawing = False
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1

    def on_mouse(self, event, x, y, flags, param):
        """Mouse olaylarını işleyen fonksiyon"""
        # Görüntü kopyasını oluştur
        current_img = self.img_copy.copy()
        
        # Tıklamayı başlat
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
        
        # Fareyi hareket ettirirken
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_x, self.end_y = x, y
        
        # Tıklamayı bırak
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x, self.end_y = x, y
            
            # ROI noktalarını kaydet (sol üst, sağ alt)
            self.roi_pts = [(min(self.start_x, self.end_x), min(self.start_y, self.end_y)),
                           (max(self.start_x, self.end_x), max(self.start_y, self.end_y))]
        
        # Seçilen alanı çiz
        if self.drawing or (self.start_x != -1 and self.end_x != -1):
            cv2.rectangle(current_img, 
                          (min(self.start_x, self.end_x), min(self.start_y, self.end_y)),
                          (max(self.start_x, self.end_x), max(self.start_y, self.end_y)),
                          (0, 255, 0), 2)
            
        # Görüntüyü göster
        cv2.imshow(self.window_name, current_img)

    def secim_yap(self, img):
        """
        Görüntüyü gösterir ve kullanıcının mouse ile seçim yapmasını sağlar.
        Seçim tamamlandığında, seçilen bölgeyi kırpar ve döndürür.
        
        Args:
            img (numpy.ndarray): Kırpılacak görüntü
            
        Returns:
            numpy.ndarray: Kırpılmış görüntü veya None (iptal edilirse)
        """
        if img is None:
            print("Hata: İşlem yapılacak bir görüntü yok!")
            return None
        
        try:
            # Görüntüyü kaydet
            self.img = img
            self.img_copy = img.copy()
            
            # Başlangıç değerlerini sıfırla
            self.roi_pts = []
            self.drawing = False
            self.start_x = -1
            self.start_y = -1
            self.end_x = -1
            self.end_y = -1
            
            # Pencereyi oluştur ve mouse olaylarını bağla
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.on_mouse)
            
            # Görüntüyü göster
            cv2.imshow(self.window_name, self.img)
            
            # Kullanıcı klavyeden bir tuşa basana kadar bekle
            print("Seçim yapmak için mouse ile bir dikdörtgen çizin.")
            print("Seçim işlemini tamamlamak için herhangi bir tuşa basın.")
            print("İptal etmek için 'ESC' tuşuna basın.")
            
            key = cv2.waitKey(0) & 0xFF
            
            # Pencereyi kapat
            cv2.destroyWindow(self.window_name)
            
            # ESC tuşuna basıldıysa iptal et
            if key == 27:  # ESC tuşu
                print("Seçim iptal edildi.")
                return None
            
            # Seçim yapıldıysa kırpma işlemini gerçekleştir
            if len(self.roi_pts) == 2:
                x1, y1 = self.roi_pts[0]
                x2, y2 = self.roi_pts[1]
                
                # Kırpma bölgesini hesapla
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                # Kırpma işlemini uygula
                cropped = kirpma(self.img, x, y, w, h)
                
                print(f"Seçilen bölge: X={x}, Y={y}, Genişlik={w}, Yükseklik={h}")
                return cropped
            else:
                print("Geçerli bir seçim yapılmadı.")
                return None
                
        except Exception as e:
            print(f"Hata: Mouse ile kırpma işlemi sırasında bir hata oluştu: {str(e)}")
            return None

def mouse_ile_kirp(img):
    """
    Mouse ile seçim yaparak görüntüyü kırpma işlemini gerçekleştirir.
    
    Args:
        img (numpy.ndarray): Kırpılacak görüntü
        
    Returns:
        numpy.ndarray: Kırpılmış görüntü veya None (iptal edilirse)
    """
    secici = MouseSeciciKirpma()
    return secici.secim_yap(img)

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
            cv2.destroyAllWindows()
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