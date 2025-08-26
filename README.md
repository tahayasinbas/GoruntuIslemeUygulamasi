# Görüntü İşleme Uygulaması (Tkinter + OpenCV)

Python ile geliştirilmiş, Tkinter tabanlı bir masaüstü görüntü işleme uygulaması. OpenCV, NumPy ve Matplotlib kullanarak; temel işlemlerden filtrelemeye, kenar bulmadan morfolojik işlemlere kadar birçok işlemi tek bir arayüzde sunar.

Ana giriş noktası: `Arayuz.py`

## İçindekiler
- [Özellikler](#özellikler)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Kullanım](#kullanım)
- [Sorun Giderme](#sorun-giderme)
- [Lisans](#lisans)

## Özellikler
- Temel İşlemler (Hafta 1)
  - Gri tonlamaya çevirme, negatif alma
  - RGB kanalları: R/G/B görüntüleme, kanallara ayırma
- Görüntü İyileştirme (Hafta 2)
  - Parlaklık ve kontrast ayarı
  - Eşikleme: basit ve adaptif
  - Histogram: görüntüleme ve eşitleme
- Geometrik Dönüşümler (Hafta 3)
  - Taşıma, aynalama (yatay/dikey/iki eksen), eğme (shear), ölçekleme, döndürme, kırpma
- Perspektif Dönüşümler (Hafta 4)
  - Perspektif düzeltme
- Filtreleme (Hafta 5)
  - Uzamsal filtreler: Ortalama, Medyan, Gauss, Konservatif, Crimmins Speckle
  - Frekans filtreleri: Fourier, Alçak/ Yüksek/ Band geçiren, Band durduran, Butterworth, Gaussian LPF/HPF, Homomorfik
- Kenar Bulma (Hafta 6)
  - Sobel, Prewitt, Roberts, Compass, Canny, Laplace, Gabor
  - Hough: Doğru ve Çember algılama
  - K-Means segmentasyon
- Morfolojik İşlemler (Hafta 7)
  - Erode, Dilate, Opening, Closing
- Dosya İşlemleri
  - Görüntü aç/kaydet/farklı kaydet
- Arayüz
  - Orijinal ve işlenmiş görüntüyü yan yana gösterme
  - Sağ panelde işlem parametreleri

## Proje Yapısı
- `Arayuz.py`: Ana Tkinter arayüzü ve menüler; uygulamayı başlatır (`if __name__ == "__main__"`).
- `Arayuz2.py`: Alternatif/deneysel arayüz; `PIL (Pillow)` ile görüntü gösterimi içerir.
- `Hafta1Ogrendiklerimiz.py`: Temel işlemler
- `Hafta2Ogrendiklerimiz.py`: Görüntü iyileştirme, histogram
- `Hafta3Ogrendiklerimiz.py`: Geometrik dönüşümler
- `Hafta4Ogrendiklerimiz.py`: Perspektif dönüşümler
- `Hafta5Ogrendiklerimiz.py`: Uzamsal ve frekans filtreleri
- `Hafta6Ogrendiklerimiz.py`: Kenar bulma, Hough, segmentasyon
- `Hafta7Ogrendiklerimiz.py`: Morfolojik işlemler
- `requirements.txt`: Bağımlılık listesi
- `__pycache__/`: Python derlenmiş dosyalar (otomatik)

## Kurulum
1) Depoyu klonlayın veya dosyaları indirin.
2) (Önerilir) Sanal ortam oluşturun ve etkinleştirin:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `./.venv/Scripts/activate`
3) Bağımlılıkları yükleyin:
   - `pip install -r requirements.txt`

`requirements.txt` içerir:
```
opencv-python>=4.8
numpy>=1.23
matplotlib>=3.7
Pillow>=10.0
```

## Çalıştırma
- Ana uygulama:
```
python Arayuz.py
```
- Alternatif arayüz (isteğe bağlı):
```
python Arayuz2.py
```

## Kullanım
- Menülerden Dosya > Aç ile bir görüntü seçin (PNG/JPG/BMP vb.).
- Sol panelde orijinal, sağda işlenmiş görüntü görünür.
- Üst menüden istediğiniz işlemi seçin; sağdaki Parametreler panelinde ayarları yapın.
- Sonucu kaydetmek için Dosya > Kaydet veya Farklı Kaydet'i kullanın.

## Sorun Giderme
- Tkinter bulunamadı: Tkinter, Windows Python dağıtımlarında hazır gelir. Özel bir Python dağıtımı kullanıyorsanız standart kurulum yapın.
- Matplotlib penceresi/çizim sorunları: Sanal ortamda kurulum yaptığınızdan emin olun ve konsolu yeniden başlatın.
- OpenCV kamera/codec hataları: Bu uygulama öncelikle statik görüntülerle çalışır; kamera gerekmez. Codec sorunlarında farklı format deneyin.
- Pillow hatası: `pip install Pillow` komutu ile tekrar kurun.

## Ekran Görüntüsü

<img width="602" height="421" alt="Ekran görüntüsü 2025-08-26 222556" src="https://github.com/user-attachments/assets/c4c336d6-7522-423b-9b03-28c47d65eb44" />
# Image Processing Application (Tkinter + OpenCV)

A desktop image processing application built with Python and Tkinter. It leverages OpenCV, NumPy, and Matplotlib to provide a wide range of operations from basic processing to filtering, edge detection, and morphological operations — all in a single UI.

Main entry point: `Arayuz.py`

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run](#run)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- Basic Operations (Week 1)
  - Convert to grayscale, negative
  - RGB channels: view R/G/B, split channels
- Image Enhancement (Week 2)
  - Adjust brightness and contrast
  - Thresholding: simple and adaptive
  - Histogram: display and equalization
- Geometric Transformations (Week 3)
  - Translation, mirroring (horizontal/vertical/both), shearing, scaling, rotation, cropping
- Perspective Transformations (Week 4)
  - Perspective correction
- Filtering (Week 5)
  - Spatial filters: Mean, Median, Gaussian, Conservative, Crimmins Speckle
  - Frequency filters: Fourier, Low/High/Band-pass, Band-stop, Butterworth, Gaussian LPF/HPF, Homomorphic
- Edge Detection (Week 6)
  - Sobel, Prewitt, Roberts, Compass, Canny, Laplace, Gabor
  - Hough: Line and Circle detection
  - K-Means segmentation
- Morphological Operations (Week 7)
  - Erode, Dilate, Opening, Closing
- File Operations
  - Open/Save/Save As image
- UI
  - Side-by-side view of original and processed images
  - Parameter panel on the right

## Project Structure
- `Arayuz.py`: Main Tkinter UI and menus; application entrypoint (`if __name__ == "__main__"`).
- `Arayuz2.py`: Alternative/experimental UI; includes display via `PIL (Pillow)`.
- `Hafta1Ogrendiklerimiz.py`: Basic operations
- `Hafta2Ogrendiklerimiz.py`: Enhancement and histogram
- `Hafta3Ogrendiklerimiz.py`: Geometric transformations
- `Hafta4Ogrendiklerimiz.py`: Perspective transformations
- `Hafta5Ogrendiklerimiz.py`: Spatial and frequency filters
- `Hafta6Ogrendiklerimiz.py`: Edge detection, Hough, segmentation
- `Hafta7Ogrendiklerimiz.py`: Morphological operations
- `requirements.txt`: Dependency list
- `__pycache__/`: Compiled Python bytecode (auto-generated)

## Setup
1) Clone the repo or download the files.
2) (Recommended) Create and activate a virtual environment:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `./.venv/Scripts/activate`
3) Install dependencies:
   - `pip install -r requirements.txt`

`requirements.txt` contains:
```
opencv-python>=4.8
numpy>=1.23
matplotlib>=3.7
Pillow>=10.0
```

## Run
- Main app:
```
python Arayuz.py
```
- Alternative UI (optional):
```
python Arayuz2.py
```

## Usage
- From the menu, choose File > Open and select an image (PNG/JPG/BMP, etc.).
- The left panel shows the original image; the right shows the processed result.
- Choose the desired operation from the top menu; adjust parameters in the right panel.
- Use File > Save or Save As to export the result.

## Troubleshooting
- Tkinter not found: Tkinter ships with standard Windows Python. If using a custom distribution, install a standard Python.
- Matplotlib display issues: Ensure you installed inside a virtual environment and restart the console.
- OpenCV camera/codec errors: The app works primarily with static images; a camera is not required. Try different file formats if codecs fail.
- Pillow error: Reinstall with `pip install Pillow`.
## Screenshots
<img width="602" height="421" alt="Ekran görüntüsü 2025-08-26 222556" src="https://github.com/user-attachments/assets/c4c336d6-7522-423b-9b03-28c47d65eb44" />



