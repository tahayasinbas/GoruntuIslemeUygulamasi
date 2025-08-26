import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspektif_duzeltme(img, kaynak_noktalar, hedef_boyut=(500, 500)):
    """
    Belirtilen 4 nokta arasındaki perspektifi düzelterek yeni bir görüntü oluşturur.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        kaynak_noktalar (numpy.ndarray): Dönüştürülecek 4 nokta (sol üst, sağ üst, sol alt, sağ alt sırasıyla)
        hedef_boyut (tuple, optional): Çıktı görüntüsünün boyutu (genişlik, yükseklik). Varsayılan (500, 500).
        
    Returns:
        numpy.ndarray: Perspektifi düzeltilmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        width, height = hedef_boyut
        
        # Hedef noktaları belirle (düzgün dikdörtgen)
        hedef_noktalar = np.float32([
            [0, 0],           # Sol üst
            [width, 0],       # Sağ üst
            [0, height],      # Sol alt
            [width, height]   # Sağ alt
        ])
        
        # Perspektif dönüşüm matrisini hesapla
        matrix = cv2.getPerspectiveTransform(kaynak_noktalar, hedef_noktalar)
        
        # Perspektif dönüşümünü uygula
        duzeltilmis_goruntu = cv2.warpPerspective(img, matrix, (width, height))
        
        return duzeltilmis_goruntu
        
    except Exception as e:
        print(f"Hata: Perspektif düzeltme işlemi uygulanırken bir hata oluştu: {str(e)}")
        return None

def kullanici_etkilesimli_perspektif_duzeltme(img, hedef_boyut=(500, 500), goruntu_adi="Görüntü"):
    """
    Kullanıcının fare ile 4 nokta seçerek görüntünün perspektifini düzeltmesini sağlar.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        hedef_boyut (tuple, optional): Çıktı görüntüsünün boyutu (genişlik, yükseklik). Varsayılan (500, 500).
        goruntu_adi (str, optional): Görüntü penceresi başlığı. Varsayılan "Görüntü".
        
    Returns:
        numpy.ndarray: Perspektifi düzeltilmiş görüntü veya None (iptal edilirse)
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        secilen_noktalar = []
        calisma_kopya = img.copy()
        
        def nokta_sec(event, x, y, flags, param):
            """Fare tıklamalarını kaydeden fonksiyon"""
            nonlocal secilen_noktalar, calisma_kopya
            
            # Tıklama olayını işle
            if event == cv2.EVENT_LBUTTONDOWN:
                # Maksimum 4 nokta seçilebilir
                if len(secilen_noktalar) < 4:
                    secilen_noktalar.append((x, y))
                    print(f"Nokta Seçildi: {x}, {y}")
                    
                    # Noktayı çiz
                    calisma_kopya = img.copy()
                    for i, (px, py) in enumerate(secilen_noktalar):
                        cv2.circle(calisma_kopya, (px, py), 5, (0, 0, 255), -1)
                        cv2.putText(calisma_kopya, f"{i+1}", (px+10, py+10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Noktaları birleştir (sıralarına göre)
                    if len(secilen_noktalar) >= 2:
                        for i in range(len(secilen_noktalar)-1):
                            pt1 = secilen_noktalar[i]
                            pt2 = secilen_noktalar[i+1]
                            cv2.line(calisma_kopya, pt1, pt2, (255, 0, 0), 2)
                    
                    # Son nokta ile ilk nokta arasında çizgi çiz (alan tamamlandıysa)
                    if len(secilen_noktalar) == 4:
                        cv2.line(calisma_kopya, secilen_noktalar[3], secilen_noktalar[0], (255, 0, 0), 2)
                    
                    # Güncel görüntüyü göster
                    cv2.imshow(pencere_adi, calisma_kopya)
                    
                    # 4 nokta seçildiyse işlemi otomatik tamamlama (isteğe bağlı)
                    if len(secilen_noktalar) == 4:
                        cv2.waitKey(1000)  # 1 saniye beklet
        
        # Görüntüyü göster ve fare olaylarını bağla
        pencere_adi = f"{goruntu_adi} - 4 nokta seçin"
        cv2.imshow(pencere_adi, img)
        cv2.setMouseCallback(pencere_adi, nokta_sec)
        
        # Kullanıcı bilgilendirmesi
        print("KULLANICI ETKİLEŞİMLİ PERSPEKTİF DÜZELTME")
        print("-------------------------------------------")
        print("Lütfen görüntü üzerinde düzeltilmesini istediğiniz 4 noktayı sırayla seçin:")
        print("1. Sol üst köşe")
        print("2. Sağ üst köşe")
        print("3. Sol alt köşe")
        print("4. Sağ alt köşe")
        print("-------------------------------------------")
        print("İşlemi tamamlamak için R tuşuna basın.")
        print("İşlemi iptal etmek için ESC tuşuna basın.")
        print("Noktaları sıfırlamak için C tuşuna basın.")
        
        while True:
            # Tuş basımlarını bekle
            key = cv2.waitKey(0) & 0xFF
            
            # ESC ile iptal
            if key == 27:  # ESC tuşu
                print("İşlem iptal edildi.")
                cv2.destroyWindow(pencere_adi)
                return None
                
            # C ile temizle
            elif key == ord('c') or key == ord('C'):
                secilen_noktalar = []
                calisma_kopya = img.copy()
                cv2.imshow(pencere_adi, calisma_kopya)
                print("Seçilen noktalar temizlendi. Tekrar seçim yapabilirsiniz.")
                
            # R ile tamamla
            elif key == ord('r') or key == ord('R'):
                # 4 nokta seçildiğinden emin ol
                if len(secilen_noktalar) == 4:
                    break
                else:
                    print(f"Lütfen 4 nokta seçin! Şu ana kadar {len(secilen_noktalar)} nokta seçildi.")
        
        # Pencereyi kapat
        cv2.destroyWindow(pencere_adi)
        
        # 4 nokta seçildiyse perspektifi düzelt
        if len(secilen_noktalar) == 4:
            kaynak_noktalar = np.float32(secilen_noktalar)
            duzeltilmis_goruntu = perspektif_duzeltme(img, kaynak_noktalar, hedef_boyut)
            
            # Sonucu göster
            sonuc_pencere = f"{goruntu_adi} - Perspektif Düzeltilmiş"
            cv2.imshow(sonuc_pencere, duzeltilmis_goruntu)
            
            print("Perspektif düzeltme işlemi tamamlandı.")
            print("Sonuç görüntüsünü kapatmak için herhangi bir tuşa basın.")
            print("Görüntüyü kaydetmek için S tuşuna basın.")
            
            # Tuş basımını bekle
            key = cv2.waitKey(0) & 0xFF
            
            # S ile kaydet
            if key == ord('s') or key == ord('S'):
                dosya_adi = f"perspektif_duzeltilmis_{goruntu_adi.replace(' ', '_').lower()}.jpg"
                cv2.imwrite(dosya_adi, duzeltilmis_goruntu)
                print(f"Görüntü başarıyla kaydedildi: {dosya_adi}")
                
            # Pencereyi kapat
            cv2.destroyWindow(sonuc_pencere)
            
            return duzeltilmis_goruntu
            
        return None
        
    except Exception as e:
        print(f"Hata: Kullanıcı etkileşimli perspektif düzeltme sırasında bir hata oluştu: {str(e)}")
        cv2.destroyAllWindows()
        return None

def manuel_perspektif_duzeltme(img, sol_ust, sag_ust, sol_alt, sag_alt, hedef_boyut=(500, 500)):
    """
    Manuel olarak belirtilen 4 nokta arasındaki perspektifi düzeltir.
    
    Args:
        img (numpy.ndarray): İşlenecek görüntü
        sol_ust (tuple): Sol üst köşe koordinatları (x, y)
        sag_ust (tuple): Sağ üst köşe koordinatları (x, y)
        sol_alt (tuple): Sol alt köşe koordinatları (x, y)
        sag_alt (tuple): Sağ alt köşe koordinatları (x, y)
        hedef_boyut (tuple, optional): Çıktı görüntüsünün boyutu (genişlik, yükseklik). Varsayılan (500, 500).
        
    Returns:
        numpy.ndarray: Perspektifi düzeltilmiş görüntü
    """
    if img is None:
        print("Hata: İşlem yapılacak bir görüntü yok!")
        return None
    
    try:
        # Kaynak noktalarını düzenle
        kaynak_noktalar = np.float32([sol_ust, sag_ust, sol_alt, sag_alt])
        
        # Perspektif düzeltme işlemini uygula
        return perspektif_duzeltme(img, kaynak_noktalar, hedef_boyut)
        
    except Exception as e:
        print(f"Hata: Manuel perspektif düzeltme sırasında bir hata oluştu: {str(e)}")
        return None

def goruntu_ac_ve_perspektif_duzelt():
    """
    Kullanıcıdan bir görüntü dosyası seçmesini ister ve perspektif düzeltme işlemini uygular.
    Dosya seçimi ve perspektif düzeltme işlemini birleştiren yardımcı fonksiyon.
    
    Returns:
        numpy.ndarray: Perspektifi düzeltilmiş görüntü veya None (iptal edilirse)
    """
    try:
        # Dosya seç
        dosya_yolu = cv2.imread("peppers.png")  # TK dosya seçimi eklenebilir
        
        if dosya_yolu is None:
            print("Dosya seçilmedi veya geçersiz!")
            return None
            
        # Görüntüyü aç
        img = cv2.imread(dosya_yolu)
        
        if img is None:
            print(f"Hata: Görüntü yüklenemedi! ({dosya_yolu})")
            return None
            
        # Perspektif düzeltme işlemini uygula
        return kullanici_etkilesimli_perspektif_duzeltme(img)
        
    except Exception as e:
        print(f"Hata: Perspektif düzeltme işlemi sırasında bir hata oluştu: {str(e)}")
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

# Örnek kullanım
if __name__ == "__main__":
    # Dosya aç ve perspektif düzeltme işlemini uygula
    img = cv2.imread("peppers.png")
    if img is not None:
        duzeltilmis = kullanici_etkilesimli_perspektif_duzeltme(img)
        
        if duzeltilmis is not None:
            # Orijinal ve düzeltilmiş görüntüleri karşılaştır
            goruntuleri_karsilastir(img, duzeltilmis)