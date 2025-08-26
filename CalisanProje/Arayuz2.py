import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Import functions from weekly lesson files
import Hafta1Ogrendiklerimiz as hafta1
import Hafta2Ogrendiklerimiz as hafta2
import Hafta3Ogrendiklerimiz as hafta3
import Hafta4Ogrendiklerimiz as hafta4
import Hafta5Ogrendiklerimiz as hafta5
import Hafta6Ogrendiklerimiz as hafta6
import Hafta7Ogrendiklerimiz as hafta7

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")
        self.root.geometry("1200x800")
        
        # Ana değişkenler
        self.original_image = None
        self.processed_image = None
        self.current_file_path = None
        
        # Ana pencere düzeni
        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
    
    def create_menu(self):
        """Ana menü çubuğunu oluşturur"""
        self.menu_bar = tk.Menu(self.root)
        
        # Dosya menüsü
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Aç", command=self.open_image)
        file_menu.add_command(label="Kaydet", command=self.save_image)
        file_menu.add_command(label="Farklı Kaydet", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Çıkış", command=self.root.quit)
        file_menu.add_command(label="Orijinale Döndür", command=self.reset_to_original)
        self.menu_bar.add_cascade(label="Dosya", menu=file_menu)
        
        # Hafta 1 - Temel İşlemler
        basic_menu = tk.Menu(self.menu_bar, tearoff=0)
        basic_menu.add_command(label="Griye Çevir", command=self.griye_cevir)
        basic_menu.add_command(label="Negatif Al", command=self.apply_negative)
        
        # RGB kanalları alt menüsü
        rgb_menu = tk.Menu(basic_menu, tearoff=0)
        rgb_menu.add_command(label="R Kanalı", command=lambda: self.show_channel("red"))
        rgb_menu.add_command(label="G Kanalı", command=lambda: self.show_channel("green"))
        rgb_menu.add_command(label="B Kanalı", command=lambda: self.show_channel("blue"))
        rgb_menu.add_command(label="Kanallara Ayır (Hepsi)", command=self.split_channels)
        basic_menu.add_cascade(label="RGB Kanalları", menu=rgb_menu)
        
        self.menu_bar.add_cascade(label="Temel İşlemler", menu=basic_menu)
        
        # Hafta 2 - Görüntü İyileştirme
        enhancement_menu = tk.Menu(self.menu_bar, tearoff=0)
        enhancement_menu.add_command(label="Parlaklık Ayarla", command=self.adjust_brightness)
        enhancement_menu.add_command(label="Kontrast Ayarla", command=self.adjust_contrast)
        
        # Eşikleme alt menüsü
        threshold_menu = tk.Menu(enhancement_menu, tearoff=0)
        threshold_menu.add_command(label="Basit Eşikleme", command=self.apply_threshold)
        threshold_menu.add_command(label="Adaptif Eşikleme", command=self.apply_adaptive_threshold)
        enhancement_menu.add_cascade(label="Eşikleme", menu=threshold_menu)
        
        # Histogram alt menüsü
        histogram_menu = tk.Menu(enhancement_menu, tearoff=0)
        histogram_menu.add_command(label="Histogram Göster", command=self.show_histogram)
        histogram_menu.add_command(label="Histogram Eşitleme", command=self.apply_histogram_equalization)
        enhancement_menu.add_cascade(label="Histogram", menu=histogram_menu)
        
        self.menu_bar.add_cascade(label="Görüntü İyileştirme", menu=enhancement_menu)
        
        # Hafta 3 - Geometrik Dönüşümler
        geometric_menu = tk.Menu(self.menu_bar, tearoff=0)
        geometric_menu.add_command(label="Taşıma", command=self.apply_translation)
        
        # Aynalama alt menüsü
        mirror_menu = tk.Menu(geometric_menu, tearoff=0)
        mirror_menu.add_command(label="Yatay Aynalama", command=self.apply_horizontal_mirror)
        mirror_menu.add_command(label="Dikey Aynalama", command=self.apply_vertical_mirror)
        mirror_menu.add_command(label="Her İki Eksende Aynalama", command=self.apply_both_axis_mirror)
        geometric_menu.add_cascade(label="Aynalama", menu=mirror_menu)
        
        geometric_menu.add_command(label="Eğme (Shearing)", command=self.apply_shearing)
        geometric_menu.add_command(label="Ölçekleme", command=self.apply_scaling)
        geometric_menu.add_command(label="Döndürme", command=self.apply_rotation)
        geometric_menu.add_command(label="Kırpma", command=self.apply_cropping)
        
        self.menu_bar.add_cascade(label="Geometrik Dönüşümler", menu=geometric_menu)
        
        # Hafta 4 - Perspektif Dönüşümler
        perspective_menu = tk.Menu(self.menu_bar, tearoff=0)
        perspective_menu.add_command(label="Perspektif Düzeltme", command=self.apply_perspective_correction)
        self.menu_bar.add_cascade(label="Perspektif Dönüşümler", menu=perspective_menu)
        
        # Hafta 5 - Filtreleme İşlemleri
        filtering_menu = tk.Menu(self.menu_bar, tearoff=0)
        
        # Uzamsal filtreler alt menüsü
        spatial_menu = tk.Menu(filtering_menu, tearoff=0)
        spatial_menu.add_command(label="Ortalama Filtre", command=self.apply_mean_filter)
        spatial_menu.add_command(label="Medyan Filtre", command=self.apply_median_filter)
        spatial_menu.add_command(label="Gauss Filtre", command=self.apply_gaussian_filter)
        spatial_menu.add_command(label="Konservatif Filtre", command=self.apply_conservative_filter)
        spatial_menu.add_command(label="Crimmins Speckle", command=self.apply_crimmins_speckle)
        filtering_menu.add_cascade(label="Uzamsal Filtreler", menu=spatial_menu)
        
        # Frekans filtreler alt menüsü
        freq_menu = tk.Menu(filtering_menu, tearoff=0)
        freq_menu.add_command(label="Fourier Dönüşümü", command=self.apply_fourier_transform)
        freq_menu.add_command(label="Alçak Geçiren Filtre", command=self.apply_low_pass_filter)
        freq_menu.add_command(label="Yüksek Geçiren Filtre", command=self.apply_high_pass_filter)
        freq_menu.add_command(label="Band Geçiren Filtre", command=self.apply_band_pass_filter)
        freq_menu.add_command(label="Band Durduran Filtre", command=self.apply_band_stop_filter)
        freq_menu.add_command(label="Butterworth Filtre", command=self.apply_butterworth_filter)
        freq_menu.add_command(label="Gaussian LPF/HPF", command=self.apply_gaussian_frequency_filter)
        freq_menu.add_command(label="Homomorfik Filtre", command=self.apply_homomorphic_filter)
        filtering_menu.add_cascade(label="Frekans Filtreleri", menu=freq_menu)
        
        self.menu_bar.add_cascade(label="Filtreleme İşlemleri", menu=filtering_menu)
        
        # Hafta 6 - Kenar Bulma
        edge_menu = tk.Menu(self.menu_bar, tearoff=0)
        edge_menu.add_command(label="Sobel", command=self.apply_sobel)
        edge_menu.add_command(label="Prewitt", command=self.apply_prewitt)
        edge_menu.add_command(label="Roberts Cross", command=self.apply_roberts_cross)
        edge_menu.add_command(label="Compass", command=self.apply_compass)
        edge_menu.add_command(label="Canny", command=self.apply_canny)
        edge_menu.add_command(label="Laplace", command=self.apply_laplace)
        edge_menu.add_command(label="Gabor", command=self.apply_gabor)
        
        # Hough dönüşümü alt menüsü
        hough_menu = tk.Menu(edge_menu, tearoff=0)
        hough_menu.add_command(label="Doğru Algılama", command=self.apply_hough_lines)
        hough_menu.add_command(label="Çember Algılama", command=self.apply_hough_circles)
        edge_menu.add_cascade(label="Hough Dönüşümü", menu=hough_menu)
        
        edge_menu.add_command(label="K-Means Segmentasyon", command=self.apply_kmeans_segmentation)
        
        self.menu_bar.add_cascade(label="Kenar Bulma", menu=edge_menu)
        
        # Hafta 7 - Morfolojik İşlemler
        morphological_menu = tk.Menu(self.menu_bar, tearoff=0)
        morphological_menu.add_command(label="Erode (Aşındırma)", command=self.apply_erosion)
        morphological_menu.add_command(label="Dilate (Genişletme)", command=self.apply_dilation)
        morphological_menu.add_command(label="Opening (Açma)", command=self.apply_opening)
        morphological_menu.add_command(label="Closing (Kapama)", command=self.apply_closing)
        
        self.menu_bar.add_cascade(label="Morfolojik İşlemler", menu=morphological_menu)
        
        # Yardım menüsü
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Hakkında", command=self.show_about)
        self.menu_bar.add_cascade(label="Yardım", menu=help_menu)
        
        self.root.config(menu=self.menu_bar)
    
    def create_main_frame(self):
        """Ana pencere içeriğini oluşturur"""
        # Ana çerçeve
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel - Görüntü gösterme alanı
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Görüntü")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # İki görüntü için canvas
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Orijinal görüntü
        self.original_frame = ttk.LabelFrame(self.canvas_frame, text="Orijinal")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_canvas = tk.Canvas(self.original_frame, bg="lightgray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # İşlenmiş görüntü
        self.processed_frame = ttk.LabelFrame(self.canvas_frame, text="İşlenmiş")
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_canvas = tk.Canvas(self.processed_frame, bg="lightgray")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Sağ panel - Parametre ayarları
        self.params_frame = ttk.LabelFrame(self.main_frame, text="Parametreler")
        self.params_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=5, ipady=5)
        
        # Parametreler içeriği
        self.params_content = ttk.Frame(self.params_frame)
        self.params_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Varsayılan olarak parametreler paneli boş
        self.clear_params_panel()
        
    def create_status_bar(self):
        """Durum çubuğunu oluşturur"""
        self.status_bar = ttk.Label(self.root, text="Hazır", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def clear_params_panel(self):
        """Parametre panelini temizler"""
        # Parametreler panelindeki tüm widget'ları temizle
        for widget in self.params_content.winfo_children():
            widget.destroy()
        
        # Boş bir etiket ekle
        ttk.Label(self.params_content, text="İşlem seçilmedi").pack(pady=10)
    
    def update_status(self, message):
        """Durum çubuğunu günceller"""
        self.status_bar["text"] = message
    
    # Dosya işlemleri
    def open_image(self):
        """Dosya açma dialogunu gösterir ve seçilen görüntüyü yükler"""
        file_path = filedialog.askopenfilename(
            title="Görüntü Dosyası Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
                ("JPEG Dosyaları", "*.jpg;*.jpeg"),
                ("PNG Dosyaları", "*.png"),
                ("BMP Dosyaları", "*.bmp"),
                ("TIFF Dosyaları", "*.tif;*.tiff"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            # Görüntüyü yükle ve göster
            self.current_file_path = file_path
            self.original_image = hafta1.goruntu_oku(file_path)
            self.processed_image = self.original_image.copy()
            self.display_images()
            self.update_status(f"Görüntü yüklendi: {os.path.basename(file_path)}")
    
    def save_image(self):
        """İşlenmiş görüntüyü kaydeder"""
        if self.processed_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek işlenmiş görüntü yok!")
            return
        
        # Mevcut dosya yolu varsa doğrudan kaydet, yoksa farklı kaydet
        if self.current_file_path:
            success = hafta1.goruntu_kaydet(self.processed_image)
            if success:
                self.update_status(f"Görüntü kaydedildi: {os.path.basename(self.current_file_path)}")
        else:
            self.save_image_as()
    
    def save_image_as(self):
        """İşlenmiş görüntüyü farklı kaydet"""
        if self.processed_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek işlenmiş görüntü yok!")
            return
        
        success = hafta1.goruntu_kaydet(self.processed_image)
        if success:
            self.update_status("Görüntü başarıyla kaydedildi.")
    
    def display_images(self):
        """Orijinal ve işlenmiş görüntüleri ekranda gösterir"""
        if self.original_image is not None:
            # Orijinal görüntüyü göster
            self.display_on_canvas(self.original_canvas, self.original_image)
        
        if self.processed_image is not None:
            # İşlenmiş görüntüyü göster
            self.display_on_canvas(self.processed_canvas, self.processed_image)
    
    def display_on_canvas(self, canvas, image):
        """Verilen canvas'a görüntüyü gösterir"""
        # Canvas'ı temizle
        canvas.delete("all")
        
        # Görüntü boyutlarını al
        h, w = image.shape[:2]
        
        # Canvas boyutlarını al
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Ölçeklendirme faktörünü hesapla
        if canvas_width <= 1:  # Canvas henüz tam oluşmamış
            canvas.update()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
        
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        scale = min(scale_w, scale_h, 1.0)  # Max ölçek 1.0
        
        # Yeni boyutları hesapla
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Görüntüyü yeniden boyutlandır
        if scale < 1.0:
            display_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_img = image.copy()
        
        # BGR'dan RGB'ye dönüştür (eğer renkli ise)
        if len(display_img.shape) == 3:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü PhotoImage formatına dönüştür
        from PIL import Image, ImageTk
        if len(display_img.shape) == 2:  # Gri tonlama
            pil_img = Image.fromarray(display_img)
        else:  # Renkli
            pil_img = Image.fromarray(display_img)
        
        # Tkinter PhotoImage oluştur
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Canvas'a ekle ve merkeze konumlandır
        img_x = max(0, (canvas_width - new_w) // 2)
        img_y = max(0, (canvas_height - new_h) // 2)
        
        # Referansı sakla (garbage collection'dan korumak için)
        canvas.image = tk_img
        
        # Canvas'a görseli yerleştir
        canvas.create_image(img_x, img_y, anchor=tk.NW, image=tk_img)
    
    # Yardım fonksiyonları
    def show_about(self):
        """Hakkında dialogunu gösterir"""
        messagebox.showinfo(
            "Hakkında",
            "Görüntü İşleme Uygulaması\n\n"
            "Bu uygulama, görüntü işleme dersi kapsamında öğrenilen tüm "
            "konuları içeren kapsamlı bir arayüz sunar.\n\n"
            "© 2023 Tüm hakları saklıdır."
        )
    
    # İşlem fonksiyonları - Hafta 1 fonksiyonlarını uygula
    def apply_negative(self):
        """Görüntünün negatifini alır"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Negatif Alma").pack(pady=5)
        ttk.Button(self.params_content, text="Uygula", 
                  command=self.execute_negative).pack(pady=10)

    def execute_negative(self):
        """Negatif alma işlemini uygular"""
        if self.original_image is None:
            return
        
        self.processed_image = hafta1.negatif_al(self.working_image.copy())
        self.display_images()
        self.update_status("Görüntünün negatifi alındı.")

    def show_channel(self, channel):
        """Belirli bir renk kanalını gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # Kanalı hazırla
        if channel == "red":
            channel_name = "Kırmızı"
        elif channel == "green":
            channel_name = "Yeşil"
        elif channel == "blue":
            channel_name = "Mavi"
        else:
            messagebox.showerror("Hata", "Geçersiz kanal seçimi!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text=f"{channel_name} Kanalı Gösterimi").pack(pady=5)
        ttk.Button(self.params_content, text="Uygula", 
                  command=lambda ch=channel: self.execute_show_channel(ch)).pack(pady=10)
              
    def execute_show_channel(self, channel):
        """Seçilen kanalı gösterme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Kanala göre işlem yap
        self.processed_image = hafta1.kanali_goster(self.working_image.copy(), channel)
        self.display_images()
        
        # Kanal ismi için düzgün metin oluştur
        if channel == "red":
            channel_name = "kırmızı"
        elif channel == "green":
            channel_name = "yeşil"
        elif channel == "blue":
            channel_name = "mavi"
        else:
            channel_name = channel
        
        self.update_status(f"Görüntünün {channel_name} kanalı gösteriliyor.")

    def split_channels(self):
        """Görüntüyü RGB kanallarına ayırır ve gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="RGB Kanallara Ayırma").pack(pady=5)
        ttk.Button(self.params_content, text="Uygula", 
                  command=self.execute_split_channels).pack(pady=10)

    def execute_split_channels(self):
        """Kanalları ayırma ve gösterme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Kanallarına ayır
        b, g, r = hafta1.kanallara_ayir(self.working_image.copy())
        
        if b is None or g is None or r is None:
            messagebox.showerror("Hata", "Kanallarına ayırma işlemi başarısız oldu!")
            return
        
        # Her kanalı ayrı pencerede göster
        # RGB görüntü için kanalları birleştir (Mavi kanal)
        blue_image = hafta1.kanali_goster(self.working_image.copy(), "blue")
        # Yeşil kanal
        green_image = hafta1.kanali_goster(self.working_image.copy(), "green")
        # Kırmızı kanal
        red_image = hafta1.kanali_goster(self.working_image.copy(), "red")
        
        # İşlenmiş görüntüye mavi kanalı ata (önizleme olarak)
        self.processed_image = blue_image
        self.display_images()
        
        # Yeni pencereler açarak diğer kanalları göster
        channel_window = tk.Toplevel(self.root)
        channel_window.title("RGB Kanalları")
        channel_window.geometry("800x600")
        
        # Her kanal için frame oluştur
        channels_frame = ttk.Frame(channel_window)
        channels_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Kırmızı kanal
        red_frame = ttk.LabelFrame(channels_frame, text="Kırmızı Kanal")
        red_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Kırmızı kanalı göster
        red_canvas = tk.Canvas(red_frame, bg="lightgray", width=250, height=250)
        red_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_on_canvas(red_canvas, red_image)
        
        # Yeşil kanal
        green_frame = ttk.LabelFrame(channels_frame, text="Yeşil Kanal")
        green_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Yeşil kanalı göster
        green_canvas = tk.Canvas(green_frame, bg="lightgray", width=250, height=250)
        green_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_on_canvas(green_canvas, green_image)
        
        # Mavi kanal
        blue_frame = ttk.LabelFrame(channels_frame, text="Mavi Kanal")
        blue_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Mavi kanalı göster
        blue_canvas = tk.Canvas(blue_frame, bg="lightgray", width=250, height=250)
        blue_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_on_canvas(blue_canvas, blue_image)
        
        # Gri tonlamalı görüntü
        gray_frame = ttk.LabelFrame(channels_frame, text="Gri Tonlama")
        gray_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Gri kanalı göster
        gray_canvas = tk.Canvas(gray_frame, bg="lightgray", width=250, height=250)
        gray_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_on_canvas(gray_canvas, hafta1.griye_cevir(self.working_image.copy()))
        
        # Grid hücrelerini eşit boyutlandır
        channels_frame.grid_columnconfigure(0, weight=1)
        channels_frame.grid_columnconfigure(1, weight=1)
        channels_frame.grid_rowconfigure(0, weight=1)
        channels_frame.grid_rowconfigure(1, weight=1)
        
        self.update_status("Görüntü kanallara ayrıldı ve gösterildi.")
    
    # Hafta 2 - Görüntü İyileştirme
    def adjust_brightness(self):
        """Parlaklık ayarları panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Parlaklık Ayarla").pack(pady=5)
        
        # Parlaklık değeri için slider
        ttk.Label(self.params_content, text="Parlaklık Değeri:").pack(pady=(10, 0))
        brightness_var = tk.IntVar(value=0)
        brightness_slider = ttk.Scale(
            self.params_content,
            from_=-100,
            to=100,
            variable=brightness_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        brightness_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=brightness_var).pack()
        
        # Hızlı modu seçeneği
        fast_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.params_content,
            text="Hızlı Mod",
            variable=fast_mode_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_brightness_adjustment(brightness_var.get(), fast_mode_var.get())
        ).pack(pady=10)

    def execute_brightness_adjustment(self, brightness_value, fast_mode=True):
        """Parlaklık ayarlama işlemini uygular"""
        if self.original_image is None:
            return
        
        # Seçilen moda göre işlevi çağır
        if fast_mode:
            self.processed_image = hafta2.parlaklik_ayarla_hizli(self.working_image.copy(), brightness_value)
        else:
            self.processed_image = hafta2.parlaklik_ayarla(self.working_image.copy(), brightness_value)
        
        self.display_images()
        self.update_status(f"Parlaklık {brightness_value} birim değiştirildi.")

    def adjust_contrast(self):
        """Kontrast ayarları panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Kontrast Ayarla").pack(pady=5)
        
        # Kontrast modu seçimi
        ttk.Label(self.params_content, text="Kontrast İşlemi:").pack(pady=(10, 5))
        mode_var = tk.StringVar(value="kontrast_ayarla")
        ttk.Radiobutton(
            self.params_content,
            text="Kontrast Faktörü (Alpha-Beta)",
            variable=mode_var,
            value="kontrast_ayarla"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Kontrast Germe (Min-Max)",
            variable=mode_var,
            value="kontrast_germe"
        ).pack(anchor=tk.W, pady=2)
        
        # Kontrast faktörü (alpha) için slider
        alpha_frame = ttk.Frame(self.params_content)
        alpha_frame.pack(pady=10, fill=tk.X)
        ttk.Label(alpha_frame, text="Kontrast (Alpha):").pack(side=tk.LEFT)
        alpha_var = tk.DoubleVar(value=1.0)
        alpha_slider = ttk.Scale(
            alpha_frame,
            from_=0.1,
            to=3.0,
            variable=alpha_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        alpha_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(alpha_frame, textvariable=alpha_var).pack(side=tk.LEFT)
        
        # Parlaklık (beta) için slider
        beta_frame = ttk.Frame(self.params_content)
        beta_frame.pack(pady=10, fill=tk.X)
        ttk.Label(beta_frame, text="Parlaklık (Beta):").pack(side=tk.LEFT)
        beta_var = tk.IntVar(value=0)
        beta_slider = ttk.Scale(
            beta_frame,
            from_=-100,
            to=100,
            variable=beta_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        beta_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(beta_frame, textvariable=beta_var).pack(side=tk.LEFT)
        
        # Histogram gösterme seçeneği
        show_histogram_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_content,
            text="Histogram Göster",
            variable=show_histogram_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_contrast_adjustment(
                mode_var.get(),
                alpha_var.get(),
                beta_var.get(),
                show_histogram_var.get()
            )
        ).pack(pady=10)

    def execute_contrast_adjustment(self, mode, alpha, beta, show_histogram):
        """Kontrast ayarlama işlemini uygular"""
        if self.original_image is None:
            return
        
        if mode == "kontrast_ayarla":
            self.processed_image = hafta2.kontrast_ayarla(self.working_image.copy(), alpha, beta)
            operation_name = f"Kontrast ayarlandı (Alpha: {alpha:.2f}, Beta: {beta})"
        else:  # kontrast_germe
            self.processed_image = hafta2.kontrast_germe(self.working_image.copy(), show_histogram)
            operation_name = "Kontrast germe uygulandı"
        
        self.display_images()
        self.update_status(operation_name)

    def apply_threshold(self):
        """Eşikleme panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Eşikleme İşlemi").pack(pady=5)
        
        # Eşik değeri için slider
        ttk.Label(self.params_content, text="Eşik Değeri:").pack(pady=(10, 0))
        threshold_var = tk.IntVar(value=127)
        threshold_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        threshold_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=threshold_var).pack()
        
        # Max değer için slider
        ttk.Label(self.params_content, text="Maksimum Değer:").pack(pady=(10, 0))
        max_val_var = tk.IntVar(value=255)
        max_val_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=max_val_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        max_val_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=max_val_var).pack()
        
        # Hızlı modu seçeneği
        fast_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.params_content,
            text="Hızlı Mod",
            variable=fast_mode_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_threshold(threshold_var.get(), max_val_var.get(), fast_mode_var.get())
        ).pack(pady=10)

    def execute_threshold(self, threshold_value, max_value, fast_mode=True):
        """Eşikleme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Seçilen moda göre işlevi çağır
        if fast_mode:
            self.processed_image = hafta2.esikleme_hizli(self.working_image.copy(), threshold_value, max_value)
        else:
            self.processed_image = hafta2.esikleme(self.working_image.copy(), threshold_value, max_value)
        
        self.display_images()
        self.update_status(f"Eşikleme uygulandı (Eşik: {threshold_value}, Max: {max_value}).")

    def apply_adaptive_threshold(self):
        """Adaptif eşikleme panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Adaptif Eşikleme").pack(pady=5)
        
        # Maksimum değer için slider
        ttk.Label(self.params_content, text="Maksimum Değer:").pack(pady=(10, 0))
        max_val_var = tk.IntVar(value=255)
        max_val_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=max_val_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        max_val_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=max_val_var).pack()
        
        # Adaptif yöntem seçimi
        ttk.Label(self.params_content, text="Adaptif Yöntem:").pack(pady=(10, 5))
        method_var = tk.IntVar(value=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        ttk.Radiobutton(
            self.params_content,
            text="Gaussian",
            variable=method_var,
            value=cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Mean",
            variable=method_var,
            value=cv2.ADAPTIVE_THRESH_MEAN_C
        ).pack(anchor=tk.W, pady=2)
        
        # Eşik tipi seçimi
        ttk.Label(self.params_content, text="Eşik Tipi:").pack(pady=(10, 5))
        thresh_type_var = tk.IntVar(value=cv2.THRESH_BINARY)
        ttk.Radiobutton(
            self.params_content,
            text="Binary",
            variable=thresh_type_var,
            value=cv2.THRESH_BINARY
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Binary Inverted",
            variable=thresh_type_var,
            value=cv2.THRESH_BINARY_INV
        ).pack(anchor=tk.W, pady=2)
        
        # Blok boyutu için slider
        ttk.Label(self.params_content, text="Blok Boyutu:").pack(pady=(10, 0))
        block_size_var = tk.IntVar(value=11)
        block_size_slider = ttk.Scale(
            self.params_content,
            from_=3,
            to=51,
            variable=block_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: block_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        block_size_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=block_size_var).pack()
        
        # C sabiti için slider
        ttk.Label(self.params_content, text="C Sabiti:").pack(pady=(10, 0))
        c_const_var = tk.IntVar(value=2)
        c_const_slider = ttk.Scale(
            self.params_content,
            from_=-10,
            to=20,
            variable=c_const_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        c_const_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=c_const_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_adaptive_threshold(
                max_val_var.get(),
                method_var.get(),
                thresh_type_var.get(),
                block_size_var.get(),
                c_const_var.get()
            )
        ).pack(pady=10)

    def execute_adaptive_threshold(self, max_value, method, thresh_type, block_size, c):
        """Adaptif eşikleme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Adaptif eşikleme uygula
        self.processed_image = hafta2.adaptif_esikleme(
            self.working_image.copy(), 
            max_deger=max_value,
            adaptif_yontem=method,
            esik_tipi=thresh_type,
            blok_boyutu=block_size,
            c=c
        )
        
        self.display_images()
        self.update_status(f"Adaptif eşikleme uygulandı (Blok: {block_size}, C: {c}).")

    def show_histogram(self):
        """Histogram gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Histogram Görüntüleme").pack(pady=5)
        
        # Görüntü seçimi
        ttk.Label(self.params_content, text="Görüntü Seçimi:").pack(pady=(10, 5))
        image_choice_var = tk.StringVar(value="original")
        ttk.Radiobutton(
            self.params_content,
            text="Orijinal Görüntü",
            variable=image_choice_var,
            value="original"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="İşlenmiş Görüntü",
            variable=image_choice_var,
            value="processed"
        ).pack(anchor=tk.W, pady=2)
        
        # Görüntü tipi seçimi
        ttk.Label(self.params_content, text="Görüntü Tipi:").pack(pady=(10, 5))
        image_type_var = tk.StringVar(value="rgb")
        ttk.Radiobutton(
            self.params_content,
            text="RGB Histogram",
            variable=image_type_var,
            value="rgb"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Gri Tonlama",
            variable=image_type_var,
            value="gray"
        ).pack(anchor=tk.W, pady=2)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Histogram Göster",
            command=lambda: self.execute_show_histogram(image_choice_var.get(), image_type_var.get())
        ).pack(pady=10)

    def execute_show_histogram(self, image_choice, image_type):
        """Histogram gösterme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Seçilen görüntüyü al
        if image_choice == "original":
            image = self.working_image.copy()
            image_name = "Orijinal"
        else:
            if self.processed_image is None:
                messagebox.showwarning("Uyarı", "İşlenmiş görüntü bulunmuyor!")
                return
            image = self.processed_image.copy()
            image_name = "İşlenmiş"
        
        # Gri tonlama seçimi yapıldıysa ve görüntü renkli ise, gri tonlamaya çevir
        if image_type == "gray" and len(image.shape) == 3:
            image = hafta1.griye_cevir(image)
        
        # Histogram penceresini oluştur
        histogram_window = tk.Toplevel(self.root)
        histogram_window.title(f"{image_name} Görüntü Histogramı")
        histogram_window.geometry("800x600")
        
        # Matplotlib figür oluştur
        fig = plt.Figure(figsize=(7, 5), dpi=100)
        
        # Histogramı hesapla ve çiz
        if len(image.shape) == 3 and image_type == "rgb":  # Renkli görüntü ve RGB histogram seçildi
            color = ('b', 'g', 'r')
            labels = ('Mavi', 'Yeşil', 'Kırmızı')
            
            ax = fig.add_subplot(111)
            for i, (col, label) in enumerate(zip(color, labels)):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=col, label=label)
            
            ax.set_title(f"{image_name} Görüntü RGB Histogramı")
            ax.legend()
        else:  # Gri tonlamalı görüntü
            # Gri tonlamalı histogram
            if len(image.shape) == 3:
                # Eğer hala RGB ise ve gri tonlama istendiyse
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            ax = fig.add_subplot(111)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black')
            ax.set_title(f"{image_name} Görüntü Gri Tonlama Histogramı")
        
        ax.set_xlabel("Piksel Değeri")
        ax.set_ylabel("Frekans")
        ax.set_xlim([0, 256])
        
        # Figürü canvas'a yerleştir
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.update_status(f"{image_name} görüntünün histogramı gösteriliyor.")

    def apply_histogram_equalization(self):
        """Histogram eşitleme panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Histogram Eşitleme").pack(pady=5)
        
        # Histogram gösterme seçeneği
        show_histogram_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_content,
            text="Histogram Karşılaştır",
            variable=show_histogram_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_histogram_equalization(show_histogram_var.get())
        ).pack(pady=10)

    def execute_histogram_equalization(self, show_histogram):
        """Histogram eşitleme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Histogram eşitleme uygula
        self.processed_image = hafta2.histogram_esitleme(self.working_image.copy(), show_histogram)
        
        self.display_images()
        self.update_status("Histogram eşitleme uygulandı.")

    # Hafta 3 - Geometrik Dönüşümler
    def apply_translation(self):
        """Taşıma işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Görüntü Taşıma").pack(pady=5)
        
        # X ekseni taşıma miktarı için slider
        ttk.Label(self.params_content, text="X Ekseninde Taşıma:").pack(pady=(10, 0))
        dx_var = tk.IntVar(value=0)
        dx_slider = ttk.Scale(
            self.params_content,
            from_=-100,
            to=100,
            variable=dx_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        dx_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=dx_var).pack()
        
        # Y ekseni taşıma miktarı için slider
        ttk.Label(self.params_content, text="Y Ekseninde Taşıma:").pack(pady=(10, 0))
        dy_var = tk.IntVar(value=0)
        dy_slider = ttk.Scale(
            self.params_content,
            from_=-100,
            to=100,
            variable=dy_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        dy_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=dy_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_translation(dx_var.get(), dy_var.get())
        ).pack(pady=10)

    def execute_translation(self, dx, dy):
        """Taşıma işlemini uygular"""
        if self.original_image is None:
            return
        
        self.processed_image = hafta3.tasima(self.working_image.copy(), dx, dy)
        self.display_images()
        self.update_status(f"Görüntü X:{dx}, Y:{dy} piksel taşındı.")

    def apply_horizontal_mirror(self):
        """Yatay aynalama uygular"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Yatay Aynalama").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Görüntüyü yatay eksende aynalar.\n(Yukarıdan aşağıya çevirir)"
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=self.execute_horizontal_mirror
        ).pack(pady=10)

    def execute_horizontal_mirror(self):
        """Yatay aynalama işlemini uygular"""
        if self.original_image is None:
            return
        
        self.processed_image = hafta3.aynalama_yatay(self.working_image.copy())
        self.display_images()
        self.update_status("Görüntü yatay eksende aynalandı.")

    def apply_vertical_mirror(self):
        """Dikey aynalama uygular"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Dikey Aynalama").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Görüntüyü dikey eksende aynalar.\n(Soldan sağa çevirir)"
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=self.execute_vertical_mirror
        ).pack(pady=10)

    def execute_vertical_mirror(self):
        """Dikey aynalama işlemini uygular"""
        if self.original_image is None:
            return
        
        self.processed_image = hafta3.aynalama_dikey(self.working_image.copy())
        self.display_images()
        self.update_status("Görüntü dikey eksende aynalandı.")

    def apply_both_axis_mirror(self):
        """Her iki eksende aynalama uygular"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Her İki Eksende Aynalama").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Görüntüyü hem yatay hem dikey eksende aynalar.\n(180 derece döndürmeye eşdeğer)"
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=self.execute_both_axis_mirror
        ).pack(pady=10)

    def execute_both_axis_mirror(self):
        """Her iki eksende aynalama işlemini uygular"""
        if self.original_image is None:
            return
        
        self.processed_image = hafta3.aynalama_her_iki_eksen(self.working_image.copy())
        self.display_images()
        self.update_status("Görüntü her iki eksende aynalandı.")

    def apply_shearing(self):
        """Eğme işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Eğme (Shearing)").pack(pady=5)
        
        # Eğme tipi seçimi
        ttk.Label(self.params_content, text="Eğme Yönü:").pack(pady=(10, 5))
        shear_type_var = tk.StringVar(value="x")
        ttk.Radiobutton(
            self.params_content,
            text="X Ekseni",
            variable=shear_type_var,
            value="x"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Y Ekseni",
            variable=shear_type_var,
            value="y"
        ).pack(anchor=tk.W, pady=2)
        
        # Eğme miktarı için slider
        ttk.Label(self.params_content, text="Eğme Miktarı:").pack(pady=(10, 0))
        shear_amount_var = tk.DoubleVar(value=0.0)
        shear_slider = ttk.Scale(
            self.params_content,
            from_=-1.0,
            to=1.0,
            variable=shear_amount_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        shear_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=shear_amount_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_shearing(shear_type_var.get(), shear_amount_var.get())
        ).pack(pady=10)

    def execute_shearing(self, shear_type, shear_amount):
        """Eğme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Eğme tipine göre fonksiyon seç
        if shear_type == "x":
            self.processed_image = hafta3.egme_x(self.working_image.copy(), shear_amount)
            axis_name = "X"
        else:  # y ekseni
            self.processed_image = hafta3.egme_y(self.working_image.copy(), shear_amount)
            axis_name = "Y"
        
        self.display_images()
        self.update_status(f"Görüntü {axis_name} ekseninde {shear_amount:.2f} birim eğildi.")

    def apply_scaling(self):
        """Ölçekleme işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Ölçekleme").pack(pady=5)
        
        # Ölçekleme faktörü için slider
        ttk.Label(self.params_content, text="Ölçek Faktörü:").pack(pady=(10, 0))
        scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(
            self.params_content,
            from_=0.1,
            to=3.0,
            variable=scale_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        scale_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=scale_var).pack()
        
        # İnterpolasyon yöntemi seçimi
        ttk.Label(self.params_content, text="İnterpolasyon Yöntemi:").pack(pady=(10, 5))
        interp_var = tk.IntVar(value=cv2.INTER_LINEAR)
        ttk.Radiobutton(
            self.params_content,
            text="Bilinear (INTER_LINEAR)",
            variable=interp_var,
            value=cv2.INTER_LINEAR
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="En Yakın Komşu (INTER_NEAREST)",
            variable=interp_var,
            value=cv2.INTER_NEAREST
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Kübik (INTER_CUBIC)",
            variable=interp_var,
            value=cv2.INTER_CUBIC
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Alan İlişkisi (INTER_AREA)",
            variable=interp_var,
            value=cv2.INTER_AREA
        ).pack(anchor=tk.W, pady=2)
        
        # Manuel ölçekleme seçeneği
        ttk.Label(self.params_content, text="Not: Manuel ölçekleme sadece küçültme için kullanılabilir").pack(pady=(10, 5))
        manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_content,
            text="Manuel Ölçekleme (Sadece Küçültme)",
            variable=manual_var
        ).pack(pady=5)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_scaling(scale_var.get(), interp_var.get(), manual_var.get())
        ).pack(pady=10)

    def execute_scaling(self, scale_factor, interpolation_method, use_manual):
        """Ölçekleme işlemini uygular"""
        if self.original_image is None:
            return
        
        if use_manual and scale_factor > 1.0:
            try:
                # Manuel ölçekleme için tam sayı faktörü kullan
                scale_factor_int = int(scale_factor)
                if scale_factor_int < 1:
                    scale_factor_int = 1
                self.processed_image = hafta3.olcekleme_manuel(self.working_image.copy(), scale_factor_int)
                operation_name = f"Manuel ölçekleme: 1/{scale_factor_int}"
            except Exception as e:
                messagebox.showerror("Hata", f"Manuel ölçekleme başarısız: {str(e)}")
                return
        else:
            # Normal ölçekleme
            self.processed_image = hafta3.olcekleme(self.working_image.copy(), scale_factor, interpolation_method)
            operation_name = f"Ölçekleme faktörü: {scale_factor:.2f}"
        
        self.display_images()
        self.update_status(operation_name)

    def apply_rotation(self):
        """Döndürme işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Görüntü Döndürme").pack(pady=5)
        
        # Döndürme açısı için slider
        ttk.Label(self.params_content, text="Döndürme Açısı (Derece):").pack(pady=(10, 0))
        angle_var = tk.DoubleVar(value=0.0)
        angle_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=360,
            variable=angle_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        angle_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=angle_var).pack()
        
        # Ölçekleme faktörü için slider
        ttk.Label(self.params_content, text="Ölçekleme Faktörü:").pack(pady=(10, 0))
        scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(
            self.params_content,
            from_=0.1,
            to=2.0,
            variable=scale_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        scale_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=scale_var).pack()
        
        # İnterpolasyon yöntemi seçimi
        ttk.Label(self.params_content, text="İnterpolasyon Yöntemi:").pack(pady=(10, 5))
        interp_var = tk.IntVar(value=cv2.INTER_LINEAR)
        ttk.Radiobutton(
            self.params_content,
            text="Bilinear (INTER_LINEAR)",
            variable=interp_var,
            value=cv2.INTER_LINEAR
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="En Yakın Komşu (INTER_NEAREST)",
            variable=interp_var,
            value=cv2.INTER_NEAREST
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Kübik (INTER_CUBIC)",
            variable=interp_var,
            value=cv2.INTER_CUBIC
        ).pack(anchor=tk.W, pady=2)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_rotation(angle_var.get(), scale_var.get(), interp_var.get())
        ).pack(pady=10)

    def execute_rotation(self, angle, scale_factor, interpolation_method):
        """Döndürme işlemini uygular"""
        if self.original_image is None:
            return
        
        self.processed_image = hafta3.dondurme(
            self.working_image.copy(), 
            angle, 
            center=None,  # Görüntü merkezini kullan
            scale=scale_factor,
            interpolation=interpolation_method
        )
        
        self.display_images()
        self.update_status(f"Görüntü {angle:.1f}° döndürüldü, ölçek: {scale_factor:.2f}")

    def apply_cropping(self):
        """Kırpma işlemi başlatır"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Görüntü Kırpma").pack(pady=5)
        
        # Kırpma seçenekleri
        ttk.Label(self.params_content, text="Kırpma Yöntemi:").pack(pady=(10, 5))
        crop_method_var = tk.StringVar(value="manual")
        ttk.Radiobutton(
            self.params_content,
            text="Manuel Kırpma (Fare ile Seçim)",
            variable=crop_method_var,
            value="manual"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Koordinatlarla Kırpma",
            variable=crop_method_var,
            value="coordinates",
            command=lambda: self.show_coordinate_inputs()
        ).pack(anchor=tk.W, pady=2)
        
        # Koordinat girişi için frame
        self.coord_frame = ttk.Frame(self.params_content)
        self.coord_frame.pack(pady=10, fill=tk.X)
        
        # X, Y, genişlik ve yükseklik değerleri için girdi alanları
        self.x_var = tk.IntVar(value=0)
        self.y_var = tk.IntVar(value=0)
        self.width_var = tk.IntVar(value=100)
        self.height_var = tk.IntVar(value=100)
        
        # Koordinat frame'ini gizle (başlangıçta)
        self.coord_frame.pack_forget()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_cropping(crop_method_var.get())
        ).pack(pady=10)

    def show_coordinate_inputs(self):
        """Koordinat girişi alanlarını gösterir"""
        # Önce frame'i temizle
        for widget in self.coord_frame.winfo_children():
            widget.destroy()
        
        # Görüntü boyutlarını al
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            # Varsayılan değerleri ayarla
            self.width_var.set(w // 2)
            self.height_var.set(h // 2)
        
        # X koordinatı
        x_frame = ttk.Frame(self.coord_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            x_frame,
            textvariable=self.x_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Y koordinatı
        y_frame = ttk.Frame(self.coord_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            y_frame,
            textvariable=self.y_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Genişlik
        width_frame = ttk.Frame(self.coord_frame)
        width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(width_frame, text="Genişlik:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            width_frame,
            textvariable=self.width_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Yükseklik
        height_frame = ttk.Frame(self.coord_frame)
        height_frame.pack(fill=tk.X, pady=2)
        ttk.Label(height_frame, text="Yükseklik:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            height_frame,
            textvariable=self.height_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Frame'i göster
        self.coord_frame.pack(pady=10, fill=tk.X)

    def execute_cropping(self, crop_method):
        """Kırpma işlemini uygular"""
        if self.original_image is None:
            return
        
        if crop_method == "manual":
            # Manuel kırpma (fare ile interaktif)
            self.update_status("Fare ile bir bölge seçin. Bekleniyor...")
            cropped_image = hafta3.mouse_ile_kirp(self.working_image.copy())
            
            if cropped_image is not None:
                self.processed_image = cropped_image
                self.display_images()
                self.update_status("Görüntü fare ile seçilen bölgeye göre kırpıldı.")
            else:
                self.update_status("Kırpma işlemi iptal edildi veya başarısız oldu.")
        else:
            # Koordinatlarla kırpma
            try:
                x = self.x_var.get()
                y = self.y_var.get()
                width = self.width_var.get()
                height = self.height_var.get()
                
                self.processed_image = hafta3.kirpma(self.working_image.copy(), x, y, width, height)
                self.display_images()
                self.update_status(f"Görüntü kırpıldı: X:{x}, Y:{y}, G:{width}, Y:{height}")
            except Exception as e:
                messagebox.showerror("Hata", f"Kırpma işlemi başarısız: {str(e)}")

    # Hafta 4 - Perspektif Dönüşümler
    def apply_perspective_correction(self):
        """Perspektif düzeltme işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Perspektif Düzeltme").pack(pady=5)
        
        # Perspektif düzeltme yöntemi seçimi
        ttk.Label(self.params_content, text="Düzeltme Yöntemi:").pack(pady=(10, 5))
        method_var = tk.StringVar(value="interactive")
        ttk.Radiobutton(
            self.params_content,
            text="İnteraktif (Fare ile Seçim)",
            variable=method_var,
            value="interactive"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Manuel (Koordinat Girişi)",
            variable=method_var,
            value="manual",
            command=lambda: self.show_perspective_coordinates()
        ).pack(anchor=tk.W, pady=2)
        
        # Koordinat girişi için frame
        self.perspective_frame = ttk.Frame(self.params_content)
        self.perspective_frame.pack(pady=10, fill=tk.X)
        
        # Hedef boyut için girdi alanları
        ttk.Label(self.params_content, text="Hedef Görüntü Boyutu:").pack(pady=(10, 5))
        
        width_height_frame = ttk.Frame(self.params_content)
        width_height_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(width_height_frame, text="Genişlik:").pack(side=tk.LEFT, padx=5)
        self.target_width_var = tk.IntVar(value=500)
        ttk.Entry(
            width_height_frame,
            textvariable=self.target_width_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(width_height_frame, text="Yükseklik:").pack(side=tk.LEFT, padx=5)
        self.target_height_var = tk.IntVar(value=500)
        ttk.Entry(
            width_height_frame,
            textvariable=self.target_height_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Koordinat frame'ini gizle (başlangıçta)
        self.perspective_frame.pack_forget()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_perspective_correction(method_var.get())
        ).pack(pady=10)

    def show_perspective_coordinates(self):
        """Perspektif düzeltme için koordinat girişi alanlarını gösterir"""
        # Önce frame'i temizle
        for widget in self.perspective_frame.winfo_children():
            widget.destroy()
        
        # Görüntü boyutlarını al
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            
            # Varsayılan değerleri ayarla (görüntünün köşeleri)
            self.sol_ust_x = tk.IntVar(value=0)
            self.sol_ust_y = tk.IntVar(value=0)
            
            self.sag_ust_x = tk.IntVar(value=w-1)
            self.sag_ust_y = tk.IntVar(value=0)
            
            self.sol_alt_x = tk.IntVar(value=0)
            self.sol_alt_y = tk.IntVar(value=h-1)
            
            self.sag_alt_x = tk.IntVar(value=w-1)
            self.sag_alt_y = tk.IntVar(value=h-1)
        else:
            # Varsayılan değerler
            self.sol_ust_x = tk.IntVar(value=0)
            self.sol_ust_y = tk.IntVar(value=0)
            
            self.sag_ust_x = tk.IntVar(value=100)
            self.sag_ust_y = tk.IntVar(value=0)
            
            self.sol_alt_x = tk.IntVar(value=0)
            self.sol_alt_y = tk.IntVar(value=100)
            
            self.sag_alt_x = tk.IntVar(value=100)
            self.sag_alt_y = tk.IntVar(value=100)
        
        # Sol üst köşe
        sol_ust_frame = ttk.LabelFrame(self.perspective_frame, text="Sol Üst Köşe")
        sol_ust_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sol_ust_frame, text="X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sol_ust_frame,
            textvariable=self.sol_ust_x,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sol_ust_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sol_ust_frame,
            textvariable=self.sol_ust_y,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Sağ üst köşe
        sag_ust_frame = ttk.LabelFrame(self.perspective_frame, text="Sağ Üst Köşe")
        sag_ust_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sag_ust_frame, text="X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sag_ust_frame,
            textvariable=self.sag_ust_x,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sag_ust_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sag_ust_frame,
            textvariable=self.sag_ust_y,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Sol alt köşe
        sol_alt_frame = ttk.LabelFrame(self.perspective_frame, text="Sol Alt Köşe")
        sol_alt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sol_alt_frame, text="X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sol_alt_frame,
            textvariable=self.sol_alt_x,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sol_alt_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sol_alt_frame,
            textvariable=self.sol_alt_y,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Sağ alt köşe
        sag_alt_frame = ttk.LabelFrame(self.perspective_frame, text="Sağ Alt Köşe")
        sag_alt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sag_alt_frame, text="X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sag_alt_frame,
            textvariable=self.sag_alt_x,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(sag_alt_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            sag_alt_frame,
            textvariable=self.sag_alt_y,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Frame'i göster
        self.perspective_frame.pack(pady=10, fill=tk.X)

    def execute_perspective_correction(self, method):
        """Perspektif düzeltme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Hedef boyutları al
        target_width = self.target_width_var.get()
        target_height = self.target_height_var.get()
        hedef_boyut = (target_width, target_height)
        
        # Seçilen yönteme göre işlem yap
        if method == "interactive":
            # İnteraktif (fare ile) perspektif düzeltme
            self.update_status("Fare ile dört köşe noktasını seçin...")
            
            # Daha iyi bir görünüm için orijinal görüntüyü büyütme
            h, w = self.original_image.shape[:2]
            scale_factor = 1.0
            
            # Çok büyük veya çok küçük görüntüler için ölçekleme yap
            if max(h, w) > 800:
                scale_factor = 800 / max(h, w)
            elif max(h, w) < 400:
                scale_factor = 400 / max(h, w)
            
            if scale_factor != 1.0:
                disp_img = cv2.resize(self.working_image.copy(), (0, 0), fx=scale_factor, fy=scale_factor)
            else:
                disp_img = self.working_image.copy()
                
            # Kullanıcıdan noktaları seçmesini iste
            duzeltilmis_goruntu = hafta4.kullanici_etkilesimli_perspektif_duzeltme(
                disp_img, 
                hedef_boyut=hedef_boyut,
                goruntu_adi="Perspektif Düzeltme"
            )
            
            if duzeltilmis_goruntu is not None:
                self.processed_image = duzeltilmis_goruntu
                self.display_images()
                self.update_status("Perspektif düzeltme işlemi tamamlandı.")
            else:
                self.update_status("Perspektif düzeltme işlemi iptal edildi veya başarısız oldu.")
        else:
            # Manuel (koordinat girişi ile) perspektif düzeltme
            try:
                # Koordinatları al
                sol_ust = (self.sol_ust_x.get(), self.sol_ust_y.get())
                sag_ust = (self.sag_ust_x.get(), self.sag_ust_y.get())
                sol_alt = (self.sol_alt_x.get(), self.sol_alt_y.get())
                sag_alt = (self.sag_alt_x.get(), self.sag_alt_y.get())
                
                # Perspektif düzeltme işlemini uygula
                self.processed_image = hafta4.manuel_perspektif_duzeltme(
                    self.working_image.copy(),
                    sol_ust, sag_ust, sol_alt, sag_alt,
                    hedef_boyut=hedef_boyut
                )
                
                if self.processed_image is not None:
                    self.display_images()
                    self.update_status("Manuel perspektif düzeltme işlemi tamamlandı.")
                else:
                    messagebox.showerror("Hata", "Perspektif düzeltme işlemi başarısız oldu!")
            except Exception as e:
                messagebox.showerror("Hata", f"Perspektif düzeltme sırasında bir hata oluştu: {str(e)}")

    # Hafta 5 - Filtreleme İşlemleri
    def apply_mean_filter(self):
        """Ortalama filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Ortalama Filtre").pack(pady=5)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=5)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=3,
            to=25,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Manuel filtreleme seçeneği
        manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_content,
            text="Manuel Filtreleme",
            variable=manual_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_mean_filter(kernel_size_var.get(), manual_var.get())
        ).pack(pady=10)

    def execute_mean_filter(self, kernel_size, manual_mode=False):
        """Ortalama filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Manuel mod veya OpenCV kullanımı seçimine göre işlev çağır
        if manual_mode:
            self.processed_image = hafta5.ortalama_filtre_manuel(self.working_image.copy(), kernel_size)
            filter_name = "Manuel ortalama filtre"
        else:
            self.processed_image = hafta5.ortalama_filtre(self.working_image.copy(), kernel_size)
            filter_name = "Ortalama filtre"
        
        self.display_images()
        self.update_status(f"{filter_name} uygulandı (Çekirdek: {kernel_size}x{kernel_size}).")

    def apply_median_filter(self):
        """Medyan filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Medyan Filtre").pack(pady=5)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=5)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=3,
            to=25,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Manuel filtreleme seçeneği
        manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_content,
            text="Manuel Filtreleme",
            variable=manual_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_median_filter(kernel_size_var.get(), manual_var.get())
        ).pack(pady=10)

    def execute_median_filter(self, kernel_size, manual_mode=False):
        """Medyan filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Medyan filtresi uygulanıyor...")
        
        # Manuel mod veya OpenCV kullanımı seçimine göre işlev çağır
        if manual_mode:
            self.processed_image = hafta5.medyan_filtre_manuel(self.working_image.copy(), kernel_size)
            filter_name = "Manuel medyan filtre"
        else:
            self.processed_image = hafta5.medyan_filtre(self.working_image.copy(), kernel_size)
            filter_name = "Medyan filtre"
        
        self.display_images()
        self.update_status(f"{filter_name} uygulandı (Çekirdek: {kernel_size}x{kernel_size}).")

    def apply_gaussian_filter(self):
        """Gauss filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Gauss Filtresi").pack(pady=5)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=5)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=3,
            to=25,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Sigma değeri için slider
        ttk.Label(self.params_content, text="Sigma Değeri:").pack(pady=(10, 0))
        sigma_var = tk.DoubleVar(value=1.0)
        sigma_slider = ttk.Scale(
            self.params_content,
            from_=0.1,
            to=5.0,
            variable=sigma_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        sigma_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=sigma_var).pack()
        
        # Manuel filtreleme seçeneği
        manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_content,
            text="Manuel Filtreleme",
            variable=manual_var
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_gaussian_filter(kernel_size_var.get(), sigma_var.get(), manual_var.get())
        ).pack(pady=10)

    def execute_gaussian_filter(self, kernel_size, sigma, manual_mode=False):
        """Gauss filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Gauss filtresi uygulanıyor...")
        
        # Manuel mod veya OpenCV kullanımı seçimine göre işlev çağır
        if manual_mode:
            self.processed_image = hafta5.gauss_filtre_manuel(self.working_image.copy(), kernel_size, sigma)
            filter_name = "Manuel Gauss filtresi"
        else:
            self.processed_image = hafta5.gauss_filtre(self.working_image.copy(), kernel_size, sigma)
            filter_name = "Gauss filtresi"
        
        self.display_images()
        self.update_status(f"{filter_name} uygulandı (Çekirdek: {kernel_size}x{kernel_size}, Sigma: {sigma:.1f}).")
    
    def apply_conservative_filter(self):
        """Konservatif filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Konservatif Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Bu filtre, piksel değerlerini sadece komşu piksel\ndeğerlerinin min ve max değerleri arasında değiştirir."
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=self.execute_conservative_filter
        ).pack(pady=10)

    def execute_conservative_filter(self):
        """Konservatif filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Konservatif filtre uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.konservatif_filtre(self.working_image.copy())
        
        self.display_images()
        self.update_status("Konservatif filtre uygulandı.")

    def apply_crimmins_speckle(self):
        """Crimmins speckle filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Crimmins Speckle Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Bu filtre, görüntüdeki benekli gürültüleri giderir."
        ).pack(pady=10)
        
        # İterasyon sayısı için slider
        ttk.Label(self.params_content, text="İterasyon Sayısı:").pack(pady=(10, 0))
        iterations_var = tk.IntVar(value=1)
        iterations_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=10,
            variable=iterations_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        iterations_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=iterations_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_crimmins_speckle(iterations_var.get())
        ).pack(pady=10)

    def execute_crimmins_speckle(self, iterations):
        """Crimmins speckle filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Crimmins speckle filtresi uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.crimmins_speckle_filtre(self.working_image.copy(), iterations)
        
        self.display_images()
        self.update_status(f"Crimmins speckle filtresi uygulandı (İterasyon: {iterations}).")

    def apply_fourier_transform(self):
        """Fourier dönüşümü gösterme panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Fourier Dönüşümü").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Görüntünün frekans spektrumunu gösterir."
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Göster",
            command=self.execute_fourier_transform
        ).pack(pady=10)

    def execute_fourier_transform(self):
        """Fourier dönüşümü gösterme işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Fourier dönüşümü hesaplanıyor...")
        
        # Fourier dönüşümünü göster
        hafta5.goruntu_fft_goster(self.working_image.copy())
        
        self.update_status("Fourier dönüşümü gösterildi.")

    def apply_low_pass_filter(self):
        """Alçak geçiren filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Alçak Geçiren Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Düşük frekansları korur, yüksek frekansları filtreler.\n(Görüntüyü yumuşatır)"
        ).pack(pady=10)
        
        # Kesme frekansı için slider
        ttk.Label(self.params_content, text="Kesme Frekansı:").pack(pady=(10, 0))
        cutoff_var = tk.IntVar(value=30)
        cutoff_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=cutoff_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        cutoff_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=cutoff_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_low_pass_filter(cutoff_var.get())
        ).pack(pady=10)

    def execute_low_pass_filter(self, cutoff_frequency):
        """Alçak geçiren filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Alçak geçiren filtre uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.alcak_geciren_filtre(self.working_image.copy(), cutoff_frequency)
        
        self.display_images()
        self.update_status(f"Alçak geçiren filtre uygulandı (Kesme frekansı: {cutoff_frequency}).")

    def apply_high_pass_filter(self):
        """Yüksek geçiren filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Yüksek Geçiren Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Yüksek frekansları korur, düşük frekansları filtreler.\n(Kenarları ve detayları vurgular)"
        ).pack(pady=10)
        
        # Kesme frekansı için slider
        ttk.Label(self.params_content, text="Kesme Frekansı:").pack(pady=(10, 0))
        cutoff_var = tk.IntVar(value=30)
        cutoff_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=cutoff_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        cutoff_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=cutoff_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_high_pass_filter(cutoff_var.get())
        ).pack(pady=10)

    def execute_high_pass_filter(self, cutoff_frequency):
        """Yüksek geçiren filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Yüksek geçiren filtre uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.yuksek_geciren_filtre(self.working_image.copy(), cutoff_frequency)
        
        self.display_images()
        self.update_status(f"Yüksek geçiren filtre uygulandı (Kesme frekansı: {cutoff_frequency}).")

    def apply_band_pass_filter(self):
        """Band geçiren filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Band Geçiren Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Belirli frekans aralığındaki frekansları korur,\ndiğerlerini filtreler."
        ).pack(pady=10)
        
        # İç yarıçap (minimum frekans) için slider
        ttk.Label(self.params_content, text="İç Yarıçap (Min. Frekans):").pack(pady=(10, 0))
        inner_radius_var = tk.IntVar(value=30)
        inner_radius_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=inner_radius_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        inner_radius_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=inner_radius_var).pack()
        
        # Dış yarıçap (maksimum frekans) için slider
        ttk.Label(self.params_content, text="Dış Yarıçap (Max. Frekans):").pack(pady=(10, 0))
        outer_radius_var = tk.IntVar(value=50)
        outer_radius_slider = ttk.Scale(
            self.params_content,
            from_=10,
            to=150,
            variable=outer_radius_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        outer_radius_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=outer_radius_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_band_pass_filter(inner_radius_var.get(), outer_radius_var.get())
        ).pack(pady=10)

    def execute_band_pass_filter(self, inner_radius, outer_radius):
        """Band geçiren filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Girilen değerlerin doğruluğunu kontrol et
        if inner_radius >= outer_radius:
            messagebox.showerror("Hata", "İç yarıçap, dış yarıçaptan küçük olmalıdır!")
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Band geçiren filtre uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.band_geciren_filtre(
            self.working_image.copy(), 
            ic_yaricap=inner_radius, 
            dis_yaricap=outer_radius
        )
        
        self.display_images()
        self.update_status(f"Band geçiren filtre uygulandı (İç yarıçap: {inner_radius}, Dış yarıçap: {outer_radius}).")

    def apply_band_stop_filter(self):
        """Band durduran filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Band Durduran Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Belirli frekans aralığındaki frekansları durdurur,\ndiğerlerini geçirir."
        ).pack(pady=10)
        
        # İç yarıçap (minimum frekans) için slider
        ttk.Label(self.params_content, text="İç Yarıçap (Min. Frekans):").pack(pady=(10, 0))
        inner_radius_var = tk.IntVar(value=30)
        inner_radius_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=inner_radius_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        inner_radius_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=inner_radius_var).pack()
        
        # Dış yarıçap (maksimum frekans) için slider
        ttk.Label(self.params_content, text="Dış Yarıçap (Max. Frekans):").pack(pady=(10, 0))
        outer_radius_var = tk.IntVar(value=50)
        outer_radius_slider = ttk.Scale(
            self.params_content,
            from_=10,
            to=150,
            variable=outer_radius_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        outer_radius_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=outer_radius_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_band_stop_filter(inner_radius_var.get(), outer_radius_var.get())
        ).pack(pady=10)

    def execute_band_stop_filter(self, inner_radius, outer_radius):
        """Band durduran filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Girilen değerlerin doğruluğunu kontrol et
        if inner_radius >= outer_radius:
            messagebox.showerror("Hata", "İç yarıçap, dış yarıçaptan küçük olmalıdır!")
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Band durduran filtre uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.band_durduran_filtre(
            self.working_image.copy(), 
            ic_yaricap=inner_radius, 
            dis_yaricap=outer_radius
        )
        
        self.display_images()
        self.update_status(f"Band durduran filtre uygulandı (İç yarıçap: {inner_radius}, Dış yarıçap: {outer_radius}).")

    def apply_butterworth_filter(self):
        """Butterworth filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Butterworth Filtre").pack(pady=5)
        
        # Filtre tipi seçimi
        ttk.Label(self.params_content, text="Filtre Tipi:").pack(pady=(10, 5))
        filter_type_var = tk.StringVar(value="low_pass")
        ttk.Radiobutton(
            self.params_content,
            text="Alçak Geçiren",
            variable=filter_type_var,
            value="low_pass"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Yüksek Geçiren",
            variable=filter_type_var,
            value="high_pass"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Band Geçiren",
            variable=filter_type_var,
            value="band_pass",
            command=lambda: self.show_band_parameters()
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Band Durduran",
            variable=filter_type_var,
            value="band_stop",
            command=lambda: self.show_band_parameters()
        ).pack(anchor=tk.W, pady=2)
        
        # Band parametreleri için frame (başlangıçta gizli)
        self.band_frame = ttk.Frame(self.params_content)
        
        # Kesme frekansı için slider
        ttk.Label(self.params_content, text="Kesme Frekansı:").pack(pady=(10, 0))
        cutoff_var = tk.IntVar(value=30)
        cutoff_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=cutoff_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        cutoff_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=cutoff_var).pack()
        
        # Derece için slider
        ttk.Label(self.params_content, text="Filtre Derecesi:").pack(pady=(10, 0))
        order_var = tk.IntVar(value=2)
        order_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=10,
            variable=order_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        order_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=order_var).pack()
        
        # İç yarıçap ve dış yarıçap için frame (Band geçiren/durduran filtreler için)
        self.band_params = {
            'inner_radius': tk.IntVar(value=20),
            'outer_radius': tk.IntVar(value=40)
        }
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_butterworth_filter(
                filter_type_var.get(),
                cutoff_var.get(),
                order_var.get()
            )
        ).pack(pady=10)

    def show_band_parameters(self):
        """Band geçiren/durduran filtreler için parametre alanlarını gösterir"""
        # Önce band frame'i temizle
        for widget in self.band_frame.winfo_children():
            widget.destroy()
        
        # İç yarıçap (minimum frekans) için slider
        ttk.Label(self.band_frame, text="İç Yarıçap:").pack(pady=(10, 0))
        inner_radius_slider = ttk.Scale(
            self.band_frame,
            from_=5,
            to=80,
            variable=self.band_params['inner_radius'],
            orient=tk.HORIZONTAL,
            length=200
        )
        inner_radius_slider.pack(pady=(0, 10))
        ttk.Label(self.band_frame, textvariable=self.band_params['inner_radius']).pack()
        
        # Dış yarıçap (maksimum frekans) için slider
        ttk.Label(self.band_frame, text="Dış Yarıçap:").pack(pady=(10, 0))
        outer_radius_slider = ttk.Scale(
            self.band_frame,
            from_=20,
            to=150,
            variable=self.band_params['outer_radius'],
            orient=tk.HORIZONTAL,
            length=200
        )
        outer_radius_slider.pack(pady=(0, 10))
        ttk.Label(self.band_frame, textvariable=self.band_params['outer_radius']).pack()
        
        # Frame'i göster
        self.band_frame.pack(pady=10, fill=tk.X)

    def execute_butterworth_filter(self, filter_type, cutoff_frequency, order):
        """Butterworth filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Filtre tipine göre farklı işlevleri çağır
        if filter_type == "low_pass":
            self.update_status(f"Butterworth alçak geçiren filtre uygulanıyor (Kesme: {cutoff_frequency}, Derece: {order})...")
            self.processed_image = hafta5.butterworth_alcak_geciren_filtre(
                self.working_image.copy(), 
                kesme_frekansi=cutoff_frequency, 
                derece=order
            )
            filter_name = "Butterworth alçak geçiren filtre"
            
        elif filter_type == "high_pass":
            self.update_status(f"Butterworth yüksek geçiren filtre uygulanıyor (Kesme: {cutoff_frequency}, Derece: {order})...")
            self.processed_image = hafta5.butterworth_yuksek_geciren_filtre(
                self.working_image.copy(), 
                kesme_frekansi=cutoff_frequency, 
                derece=order
            )
            filter_name = "Butterworth yüksek geçiren filtre"
            
        elif filter_type == "band_pass":
            inner_radius = self.band_params['inner_radius'].get()
            outer_radius = self.band_params['outer_radius'].get()
            
            # Değer kontrolü
            if inner_radius >= outer_radius:
                messagebox.showerror("Hata", "İç yarıçap, dış yarıçaptan küçük olmalıdır!")
                return
                
            self.update_status(f"Butterworth band geçiren filtre uygulanıyor...")
            self.processed_image = hafta5.butterworth_band_geciren_filtre(
                self.working_image.copy(), 
                ic_yaricap=inner_radius, 
                dis_yaricap=outer_radius, 
                derece=order
            )
            filter_name = f"Butterworth band geçiren filtre (İç: {inner_radius}, Dış: {outer_radius}, Derece: {order})"
            
        elif filter_type == "band_stop":
            inner_radius = self.band_params['inner_radius'].get()
            outer_radius = self.band_params['outer_radius'].get()
            
            # Değer kontrolü
            if inner_radius >= outer_radius:
                messagebox.showerror("Hata", "İç yarıçap, dış yarıçaptan küçük olmalıdır!")
                return
                
            self.update_status(f"Butterworth band durduran filtre uygulanıyor...")
            self.processed_image = hafta5.butterworth_band_durduran_filtre(
                self.working_image.copy(), 
                ic_yaricap=inner_radius, 
                dis_yaricap=outer_radius, 
                derece=order
            )
            filter_name = f"Butterworth band durduran filtre (İç: {inner_radius}, Dış: {outer_radius}, Derece: {order})"
        
        self.display_images()
        self.update_status(f"{filter_name} uygulandı.")

    def apply_gaussian_frequency_filter(self):
        """Gaussian frekans filtresi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Gaussian Frekans Filtresi").pack(pady=5)
        
        # Filtre tipi seçimi
        ttk.Label(self.params_content, text="Filtre Tipi:").pack(pady=(10, 5))
        filter_type_var = tk.StringVar(value="low_pass")
        ttk.Radiobutton(
            self.params_content,
            text="Alçak Geçiren",
            variable=filter_type_var,
            value="low_pass"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Yüksek Geçiren",
            variable=filter_type_var,
            value="high_pass"
        ).pack(anchor=tk.W, pady=2)
        
        # Kesme frekansı için slider
        ttk.Label(self.params_content, text="Kesme Frekansı:").pack(pady=(10, 0))
        cutoff_var = tk.IntVar(value=30)
        cutoff_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=cutoff_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        cutoff_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=cutoff_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_gaussian_frequency_filter(filter_type_var.get(), cutoff_var.get())
        ).pack(pady=10)

    def execute_gaussian_frequency_filter(self, filter_type, cutoff_frequency):
        """Gaussian frekans filtresi işlemini uygular"""
        if self.original_image is None:
            return
        
        # Filtre tipine göre farklı işlevleri çağır
        if filter_type == "low_pass":
            self.update_status(f"Gaussian alçak geçiren filtre uygulanıyor (Kesme: {cutoff_frequency})...")
            self.processed_image = hafta5.gaussian_alcak_geciren_filtre(
                self.working_image.copy(), 
                kesme_frekansi=cutoff_frequency
            )
            filter_name = "Gaussian alçak geçiren filtre"
            
        elif filter_type == "high_pass":
            self.update_status(f"Gaussian yüksek geçiren filtre uygulanıyor (Kesme: {cutoff_frequency})...")
            self.processed_image = hafta5.gaussian_yuksek_geciren_filtre(
                self.working_image.copy(), 
                kesme_frekansi=cutoff_frequency
            )
            filter_name = "Gaussian yüksek geçiren filtre"
        
        self.display_images()
        self.update_status(f"{filter_name} uygulandı (Kesme frekansı: {cutoff_frequency}).")

    def apply_homomorphic_filter(self):
        """Homomorfik filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Homomorfik Filtre").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Aydınlatma farklılıklarını azaltır ve kontrastı artırır."
        ).pack(pady=10)
        
        # Kesme frekansı için slider
        ttk.Label(self.params_content, text="Kesme Frekansı:").pack(pady=(10, 0))
        cutoff_var = tk.IntVar(value=30)
        cutoff_slider = ttk.Scale(
            self.params_content,
            from_=5,
            to=100,
            variable=cutoff_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        cutoff_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=cutoff_var).pack()
        
        # Düşük frekans kazancı için slider
        ttk.Label(self.params_content, text="Düşük Frekans Kazancı:").pack(pady=(10, 0))
        low_gain_var = tk.DoubleVar(value=0.5)
        low_gain_slider = ttk.Scale(
            self.params_content,
            from_=0.1,
            to=1.0,
            variable=low_gain_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        low_gain_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=low_gain_var).pack()
        
        # Yüksek frekans kazancı için slider
        ttk.Label(self.params_content, text="Yüksek Frekans Kazancı:").pack(pady=(10, 0))
        high_gain_var = tk.DoubleVar(value=2.0)
        high_gain_slider = ttk.Scale(
            self.params_content,
            from_=1.0,
            to=3.0,
            variable=high_gain_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        high_gain_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=high_gain_var).pack()
        
        # Keskinlik kontrolü için slider
        ttk.Label(self.params_content, text="Keskinlik Kontrolü:").pack(pady=(10, 0))
        c_var = tk.IntVar(value=1)
        c_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=5,
            variable=c_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        c_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=c_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_homomorphic_filter(
                cutoff_var.get(),
                low_gain_var.get(),
                high_gain_var.get(),
                c_var.get()
            )
        ).pack(pady=10)

    def execute_homomorphic_filter(self, cutoff_frequency, low_gain, high_gain, c):
        """Homomorfik filtre işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Homomorfik filtre uygulanıyor...")
        
        # Filtreyi uygula
        self.processed_image = hafta5.homomorfik_filtre(
            self.working_image.copy(), 
            d0=cutoff_frequency, 
            h_l=low_gain, 
            h_h=high_gain, 
            c=c
        )
        
        self.display_images()
        self.update_status(f"Homomorfik filtre uygulandı (Kesme: {cutoff_frequency}, DK: {low_gain}, YK: {high_gain}, Keskinlik: {c}).")

    # Hafta 6 - Kenar Bulma
    def apply_sobel(self):
        """Sobel kenar bulma panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Sobel Kenar Bulma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Gradyan tabanlı kenar algılama algoritması.\nX ve Y yönündeki kenarları ayrı ayrı veya\ntoplam olarak tespit edebilir."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=7,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Çıktı modu seçimi
        ttk.Label(self.params_content, text="Çıktı Modu:").pack(pady=(10, 5))
        output_mode_var = tk.StringVar(value="magnitude")
        ttk.Radiobutton(
            self.params_content,
            text="X Yönü",
            variable=output_mode_var,
            value="x"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Y Yönü",
            variable=output_mode_var,
            value="y"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Toplam (Magnitude)",
            variable=output_mode_var,
            value="magnitude"
        ).pack(anchor=tk.W, pady=2)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_sobel(kernel_size_var.get(), output_mode_var.get())
        ).pack(pady=10)

    def execute_sobel(self, kernel_size, output_mode):
        """Sobel kenar bulma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Sobel kenar bulma uygulanıyor...")
        
        # Sobel kenar bulma işlemini uygula
        sobel_x, sobel_y, sobel_magnitude = hafta6.sobel_kenar_bulma(self.working_image.copy(), kernel_size)
        
        # Seçilen moda göre çıktıyı belirle
        if output_mode == "x":
            self.processed_image = sobel_x
            mode_name = "X yönü"
        elif output_mode == "y":
            self.processed_image = sobel_y
            mode_name = "Y yönü"
        else:  # magnitude
            self.processed_image = sobel_magnitude
            mode_name = "toplam (magnitude)"
        
        self.display_images()
        self.update_status(f"Sobel kenar bulma uygulandı (Çekirdek: {kernel_size}x{kernel_size}, Mod: {mode_name}).")

    def apply_prewitt(self):
        """Prewitt kenar bulma panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Prewitt Kenar Bulma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Gradyan tabanlı kenar algılama algoritması.\nX ve Y yönündeki kenarları ayrı ayrı veya\ntoplam olarak tespit edebilir."
        ).pack(pady=10)
        
        # Çıktı modu seçimi
        ttk.Label(self.params_content, text="Çıktı Modu:").pack(pady=(10, 5))
        output_mode_var = tk.StringVar(value="magnitude")
        ttk.Radiobutton(
            self.params_content,
            text="X Yönü",
            variable=output_mode_var,
            value="x"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Y Yönü",
            variable=output_mode_var,
            value="y"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Toplam (Magnitude)",
            variable=output_mode_var,
            value="magnitude"
        ).pack(anchor=tk.W, pady=2)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_prewitt(output_mode_var.get())
        ).pack(pady=10)

    def execute_prewitt(self, output_mode):
        """Prewitt kenar bulma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Prewitt kenar bulma uygulanıyor...")
        
        # Prewitt kenar bulma işlemini uygula
        prewitt_x, prewitt_y, prewitt_magnitude = hafta6.prewitt_kenar_bulma(self.working_image.copy())
        
        # Seçilen moda göre çıktıyı belirle
        if output_mode == "x":
            self.processed_image = prewitt_x
            mode_name = "X yönü"
        elif output_mode == "y":
            self.processed_image = prewitt_y
            mode_name = "Y yönü"
        else:  # magnitude
            self.processed_image = prewitt_magnitude
            mode_name = "toplam (magnitude)"
        
        self.display_images()
        self.update_status(f"Prewitt kenar bulma uygulandı (Mod: {mode_name}).")

    def apply_roberts_cross(self):
        """Roberts Cross kenar bulma panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Roberts Cross Kenar Bulma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="2x2 çekirdekler kullanarak çapraz gradyanları hesaplar.\nKüçük çekirdek boyutu sayesinde kenarları\ndaha keskin tespit eder."
        ).pack(pady=10)
        
        # Çıktı modu seçimi
        ttk.Label(self.params_content, text="Çıktı Modu:").pack(pady=(10, 5))
        output_mode_var = tk.StringVar(value="magnitude")
        ttk.Radiobutton(
            self.params_content,
            text="X Yönü",
            variable=output_mode_var,
            value="x"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Y Yönü",
            variable=output_mode_var,
            value="y"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            self.params_content,
            text="Toplam (Magnitude)",
            variable=output_mode_var,
            value="magnitude"
        ).pack(anchor=tk.W, pady=2)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_roberts_cross(output_mode_var.get())
        ).pack(pady=10)

    def execute_roberts_cross(self, output_mode):
        """Roberts Cross kenar bulma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Roberts Cross kenar bulma uygulanıyor...")
        
        # Roberts Cross kenar bulma işlemini uygula
        roberts_x, roberts_y, roberts_magnitude = hafta6.roberts_cross_kenar_bulma(self.working_image.copy())
        
        # Seçilen moda göre çıktıyı belirle
        if output_mode == "x":
            self.processed_image = roberts_x
            mode_name = "X yönü"
        elif output_mode == "y":
            self.processed_image = roberts_y
            mode_name = "Y yönü"
        else:  # magnitude
            self.processed_image = roberts_magnitude
            mode_name = "toplam (magnitude)"
        
        self.display_images()
        self.update_status(f"Roberts Cross kenar bulma uygulandı (Mod: {mode_name}).")

    def apply_compass(self):
        """Compass kenar bulma panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Compass Kenar Bulma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Farklı yönlerdeki gradyanları hesaplayarak\nen güçlü kenarları bulur. Doğu, batı, kuzey, güney,\nkuzeydoğu, kuzeybatı, güneydoğu ve\ngüneybatı yönlerindeki kenarları tespit eder."
        ).pack(pady=10)
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=self.execute_compass
        ).pack(pady=10)

    def execute_compass(self):
        """Compass kenar bulma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Compass kenar bulma uygulanıyor...")
        
        # Compass kenar bulma işlemini uygula
        self.processed_image = hafta6.compass_kenar_bulma(self.working_image.copy())
        
        self.display_images()
        self.update_status("Compass kenar bulma uygulandı.")

    def apply_canny(self):
        """Canny kenar bulma panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Canny Kenar Bulma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Çok aşamalı kenar tespit algoritması.\nGürültü azaltma, gradyan hesaplama,\nnon-maximum suppression ve\nikili eşikleme aşamalarını içerir."
        ).pack(pady=10)
        
        # Alt eşik değeri için slider
        ttk.Label(self.params_content, text="Alt Eşik:").pack(pady=(10, 0))
        min_threshold_var = tk.IntVar(value=50)
        min_threshold_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=min_threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        min_threshold_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=min_threshold_var).pack()
        
        # Üst eşik değeri için slider
        ttk.Label(self.params_content, text="Üst Eşik:").pack(pady=(10, 0))
        max_threshold_var = tk.IntVar(value=150)
        max_threshold_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=max_threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        max_threshold_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=max_threshold_var).pack()
        
        # Aperture boyutu için combobox
        ttk.Label(self.params_content, text="Aperture Boyutu:").pack(pady=(10, 5))
        aperture_size_var = tk.IntVar(value=3)
        aperture_sizes = [3, 5, 7]
        aperture_combobox = ttk.Combobox(
            self.params_content,
            textvariable=aperture_size_var,
            values=aperture_sizes,
            state='readonly',
            width=5
        )
        aperture_combobox.pack(pady=(0, 10))
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_canny(
                min_threshold_var.get(),
                max_threshold_var.get(),
                aperture_size_var.get()
            )
        ).pack(pady=10)

    def execute_canny(self, min_threshold, max_threshold, aperture_size):
        """Canny kenar bulma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Girilen değerlerin doğruluğunu kontrol et
        if min_threshold >= max_threshold:
            messagebox.showerror("Hata", "Alt eşik, üst eşikten küçük olmalıdır!")
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Canny kenar bulma uygulanıyor...")
        
        # Canny kenar bulma işlemini uygula
        self.processed_image = hafta6.canny_kenar_bulma(
            self.working_image.copy(),
            alt_esik=min_threshold,
            ust_esik=max_threshold,
            aperture_size=aperture_size
        )
        
        self.display_images()
        self.update_status(f"Canny kenar bulma uygulandı (Alt Eşik: {min_threshold}, Üst Eşik: {max_threshold}, Aperture: {aperture_size}).")

    def apply_laplace(self):
        """Laplace kenar bulma panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Laplace Kenar Bulma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Laplacian of Gaussian (LoG) filtresi ile kenarları tespit eder."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=5)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=3,
            to=25,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Sigma değeri için slider
        ttk.Label(self.params_content, text="Sigma Değeri:").pack(pady=(10, 0))
        sigma_var = tk.DoubleVar(value=1.0)
        sigma_slider = ttk.Scale(
            self.params_content,
            from_=0.1,
            to=5.0,
            variable=sigma_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        sigma_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=sigma_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_laplace(kernel_size_var.get(), sigma_var.get())
        ).pack(pady=10)

    def execute_laplace(self, kernel_size, sigma):
        """Laplace kenar bulma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Laplace kenar bulma uygulanıyor...")
        
        # Görüntüyü gri tonlamaya çevir
        if len(self.original_image.shape) > 2:
            gray = cv2.cvtColor(self.working_image.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = self.working_image.copy()
        
        # Gürültü azaltmak için önce Gaussian filtre uygula
        if sigma > 0:
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        
        # Laplace filtresini uygula
        self.processed_image = hafta6.laplace_kenar_bulma(gray, ksize=kernel_size)
        self.display_images()
        self.update_status(f"Laplace kenar algılama filtresi uygulandı (kernel={kernel_size})")

    def apply_gabor(self):
        """Gabor filtre panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Gabor Filtresi").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Gabor filtresi, kenarları ve detayları vurgular."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=5)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=3,
            to=25,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Sigma değeri için slider
        ttk.Label(self.params_content, text="Sigma Değeri:").pack(pady=(10, 0))
        sigma_var = tk.DoubleVar(value=1.0)
        sigma_slider = ttk.Scale(
            self.params_content,
            from_=0.1,
            to=5.0,
            variable=sigma_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        sigma_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=sigma_var).pack()
        
        # Theta açısı için slider
        ttk.Label(self.params_content, text="Theta Açısı:").pack(pady=(10, 0))
        theta_var = tk.DoubleVar(value=0.0)
        theta_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=360,
            variable=theta_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        theta_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=theta_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_gabor(kernel_size_var.get(), sigma_var.get(), theta_var.get())
        ).pack(pady=10)

    def execute_gabor(self, kernel_size, sigma, theta):
        """Gabor filtresini uygular"""
        if self.original_image is None:
            return
        
        # Görüntüyü gri tonlamaya çevir
        if len(self.original_image.shape) > 2:
            gray = cv2.cvtColor(self.working_image.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = self.working_image.copy()
        
        # Gabor filtresini uygula
        self.processed_image = hafta6.gabor_filtre(gray, kernel_size=kernel_size, sigma=sigma, theta=theta)
        self.display_images()
        self.update_status(f"Gabor filtresi uygulandı (kernel={kernel_size}, sigma={sigma:.1f}, theta={theta:.2f})")

    def apply_hough_lines(self):
        """Hough çizgi algılama panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Hough Çizgi Algılama").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Doğru ve çemberleri tespit eder."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=7,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Theta açısı için slider
        ttk.Label(self.params_content, text="Theta Açısı:").pack(pady=(10, 0))
        theta_var = tk.DoubleVar(value=0.0)
        theta_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=360,
            variable=theta_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        theta_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=theta_var).pack()
        
        # Eşik değeri için slider
        ttk.Label(self.params_content, text="Eşik Değeri:").pack(pady=(10, 0))
        threshold_var = tk.IntVar(value=50)
        threshold_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        threshold_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=threshold_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_hough_lines(kernel_size_var.get(), theta_var.get(), threshold_var.get())
        ).pack(pady=10)

    def execute_hough_lines(self, kernel_size, theta, threshold):
        """Hough çizgi algılama işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Hough çizgi algılama uygulanıyor...")
        
        # Hough çizgi algılama işlemini uygula
        lines = hafta7.hough_cizgi_algilama(self.working_image.copy(), kernel_size, theta, threshold)
        
        self.processed_image = hafta7.cizgi_cizdir(self.working_image.copy(), lines)
        self.display_images()
        self.update_status("Hough çizgi algılama işlemi tamamlandı.")

    def apply_hough_circles(self):
        """Hough çember algılama panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Hough Çember Algılama").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Çemberleri tespit eder."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=7,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Eşik değeri için slider
        ttk.Label(self.params_content, text="Eşik Değeri:").pack(pady=(10, 0))
        threshold_var = tk.IntVar(value=100)
        threshold_slider = ttk.Scale(
            self.params_content,
            from_=0,
            to=255,
            variable=threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        threshold_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=threshold_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_hough_circles(kernel_size_var.get(), threshold_var.get())
        ).pack(pady=10)

    def execute_hough_circles(self, kernel_size, threshold):
        """Hough çember algılama işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Hough çember algılama uygulanıyor...")
        
        # Hough çember algılama işlemini uygula
        circles = hafta7.hough_cember_algilama(self.working_image.copy(), kernel_size, threshold)
        
        self.processed_image = hafta7.cember_cizdir(self.working_image.copy(), circles)
        self.display_images()
        self.update_status("Hough çember algılama işlemi tamamlandı.")

    def apply_kmeans_segmentation(self):
        """K-means segmentasyon panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="K-means Segmentasyon").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Görüntüyü k-means algoritması ile segmentlere ayırır."
        ).pack(pady=10)
        
        # Küme sayısı için slider
        ttk.Label(self.params_content, text="Küme Sayısı:").pack(pady=(10, 0))
        k_var = tk.IntVar(value=3)
        k_slider = ttk.Scale(
            self.params_content,
            from_=2,
            to=10,
            variable=k_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: k_var.set(int(float(val)))
        )
        k_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=k_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_kmeans_segmentation(k_var.get())
        ).pack(pady=10)

    def execute_kmeans_segmentation(self, k):
        """K-means segmentasyon işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("K-means segmentasyon uygulanıyor...")
        
        # K-means segmentasyon işlemini uygula
        _, self.processed_image, _ = hafta6.kmeans_segmentation(
            self.working_image.copy(),
            k=k
        )
        
        self.display_images()
        self.update_status(f"K-means segmentasyon işlemi tamamlandı (Küme sayısı: {k}).")

    # Hafta 7 - Morfolojik İşlemler
    def apply_erosion(self):
        """Aşındırma işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Aşındırma").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Piksel değerlerini sadece komşu piksel\ndeğerlerinin min değerine ayarlar."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=7,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_erosion(kernel_size_var.get())
        ).pack(pady=10)

    def execute_erosion(self, kernel_size):
        """Aşındırma işlemini uygular"""
        if self.original_image is None:
            return
        
        # İşlem yapılacak görüntüyü belirle
        if self.processed_image is not None:
            # İşlem yapılacak resim kopya olmalı
            working_image = self.processed_image.copy()
        else:
            # İşlenmiş resim yoksa orijinal resmi kullan
            working_image = self.working_image.copy()
        
        # Yükleme durumunu güncelle
        self.update_status("Aşındırma işlemi uygulanıyor...")
        
        # Aşındırma işlemini uygula (Hafta7'deki fonksiyon adı erode_islem)
        result_image = hafta7.erode_islem(working_image, kernel_size=kernel_size)
        
        # Sonucu işlenmiş görüntüye ata
        self.processed_image = result_image
        
        self.display_images()
        self.update_status(f"Aşındırma işlemi tamamlandı (kernel boyutu: {kernel_size}x{kernel_size}).")

    def apply_dilation(self):
        """Genişletme işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Genişletme").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Piksel değerlerini sadece komşu piksel\ndeğerlerinin max değerine ayarlar."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=7,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_dilation(kernel_size_var.get())
        ).pack(pady=10)

    def execute_dilation(self, kernel_size):
        """Genişletme işlemini uygular"""
        if self.original_image is None:
            return
        
        # İşlem yapılacak görüntüyü belirle
        if self.processed_image is not None:
            # İşlem yapılacak resim kopya olmalı
            working_image = self.processed_image.copy()
        else:
            # İşlenmiş resim yoksa orijinal resmi kullan
            working_image = self.working_image.copy()
        
        # Yükleme durumunu güncelle
        self.update_status("Genişletme uygulanıyor...")
        
        # Genişletme işlemini uygula
        result_image = hafta7.dilate_islem(working_image, kernel_size=kernel_size)
        
        # Sonucu işlenmiş görüntüye ata
        self.processed_image = result_image
        
        self.display_images()
        self.update_status(f"Genişletme uygulandı (Çekirdek: {kernel_size}x{kernel_size}).")

    def apply_opening(self):
        """Açma işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Açma (Opening)").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Önce aşındırma sonra genişletme işlemi uygular.\nKüçük nesneleri kaldırır ve gürültüyü temizler."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=15,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_opening(kernel_size_var.get())
        ).pack(pady=10)

    def execute_opening(self, kernel_size):
        """Açma işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Açma işlemi uygulanıyor...")
        
        # Açma işlemini uygula
        self.processed_image = hafta7.opening_islem(self.working_image.copy(), kernel_size=kernel_size)
        self.display_images()
        self.update_status(f"Açma işlemi tamamlandı (kernel boyutu: {kernel_size}x{kernel_size}).")

    def apply_closing(self):
        """Kapama işlemi panelini gösterir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem parametreleri panelini oluştur
        self.clear_params_panel()
        ttk.Label(self.params_content, text="Kapama (Closing)").pack(pady=5)
        ttk.Label(
            self.params_content, 
            text="Önce genişletme sonra aşındırma işlemi uygular.\nKüçük boşlukları kapatır ve nesneleri birleştirir."
        ).pack(pady=10)
        
        # Çekirdek boyutu için slider
        ttk.Label(self.params_content, text="Çekirdek Boyutu:").pack(pady=(10, 0))
        kernel_size_var = tk.IntVar(value=3)
        kernel_slider = ttk.Scale(
            self.params_content,
            from_=1,
            to=15,
            variable=kernel_size_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda val: kernel_size_var.set(int(float(val)) // 2 * 2 + 1)  # Tek sayı olması için
        )
        kernel_slider.pack(pady=(0, 10))
        ttk.Label(self.params_content, textvariable=kernel_size_var).pack()
        
        # Uygula butonu
        ttk.Button(
            self.params_content,
            text="Uygula",
            command=lambda: self.execute_closing(kernel_size_var.get())
        ).pack(pady=10)

    def execute_closing(self, kernel_size):
        """Kapama işlemini uygular"""
        if self.original_image is None:
            return
        
        # Yükleme durumunu güncelle
        self.update_status("Kapama işlemi uygulanıyor...")
        
        # Kapama işlemini uygula
        self.processed_image = hafta7.closing_islem(self.working_image.copy(), kernel_size=kernel_size)
        self.display_images()
        self.update_status(f"Kapama işlemi tamamlandı (kernel boyutu: {kernel_size}x{kernel_size}).")

    def griye_cevir(self):
        """Görüntüyü gri tonlamaya çevirir"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlem yapılacak görüntüyü belirle
        if self.processed_image is not None:
            # İşlem yapılacak resim kopya olmalı
            working_image = self.processed_image.copy()
        else:
            # İşlenmiş resim yoksa orijinal resmi kullan
            working_image = self.working_image.copy()
        
        # Görüntüyü gri tonlamaya çevir
        result_image = hafta1.griye_cevir(working_image)
        
        # Sonucu işlenmiş görüntüye ata
        self.processed_image = result_image
        
        self.display_images()
        self.update_status("Görüntü gri tonlamaya çevrildi.")

    def reset_to_original(self):
        """Görüntüyü orijinal haline döndürür"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        # İşlenmiş görüntüyü orijinal görüntüye geri döndür
        self.processed_image = self.working_image.copy()
        self.display_images()
        self.update_status("Görüntü orijinal haline döndürüldü.")

    @property
    def working_image(self):
        """Üzerinde çalışılacak görüntüyü döndürür"""
        return self.processed_image if self.processed_image is not None else self.original_image

# Uygulamayı başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

