# Astro PSF Detection

![astro_psf_detection_banner](https://via.placeholder.com/1200x400.png?text=Astro+PSF+Detection)

Astro PSF Detection, astronomik FITS görüntülerindeki kaynakları (yıldızlar, galaksiler ve uydular) tespit etmek ve sınıflandırmak için derin öğrenme tabanlı bir yaklaşımdır. Bu proje, teleskop verilerinden gelen PSF (Point Spread Function) analizine odaklanarak, farklı kaynakların otomatik tespit ve sınıflandırılmasını sağlar. PyTorch ve dağıtılmış eğitim desteği ile büyük ölçekli veri kümeleri üzerinde hızlı ve verimli eğitim sunar.

## İçindekiler

- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#iletişim)

## Özellikler

- **FITS Görüntü İşleme:** Ham FITS görüntü dosyalarını işler ve analiz için hazır hale getirir.
- **Derin Öğrenme ile Sınıflandırma:** CNN tabanlı model kullanarak yıldız, galaksi ve uydu gibi kaynakları sınıflandırır.
- **PSF Analizi:** Optik sistemlerin noktaları nasıl görüntülediğini analiz eder.
- **Otomatik Tespit ve Koordinat Belirleme:** Kaynakların pozisyonunu ve türünü otomatik olarak belirler.
- **Çizgi Tespiti:** Uydu çizgilerini tespit eder ve koordinatlarını belirler.
- **Particle Image Velocimetry (PIV) Analizi:** Hız vektörleri ve yönelim analizi yapar.
- **Paralel Eğitim Desteği:** Dağıtılmış eğitim ile büyük veri setlerini hızlı bir şekilde eğitir.
- **Önbellekte Bulunan Ağırlıklar:** COCO veya ImageNet üzerinde eğitilmiş modellerden yararlanarak daha hızlı başlangıç.

## Gereksinimler

Bu proje, Python 3.7+ sürümü ve aşağıdaki Python paketlerini gerektirir:

- `torch==2.2.0`
- `torchvision==0.17.0`
- `numpy`
- `matplotlib`
- `astropy`
- `tqdm`
- `scikit-learn`
- `opencv-python-headless`
- `pandas`
- `pyyaml`

## Kurulum

### 1. Gerekli Paketleri Yükleyin

Gerekli Python paketlerini yüklemek için aşağıdaki komutu kullanın:

```bash
pip install -r requirements.txt
```

### 2. Projeyi Klonlayın

Proje deposunu yerel makinenize klonlayın:

```bash
git clone https://github.com/mmustafakapici/astro_psf_detection.git
cd astro_psf_detection
```

### 3. Veri İndirme ve Hazırlık

**Veri İndirme:** Eğitim ve doğrulama veri setlerini indirmek için `src/download_data.py` betiğini çalıştırın:

```bash
python src/download_test_data.py
```

**Veri Ön İşleme:** `notebooks/data_preprocessing.ipynb` dosyasını açarak veri ön işleme adımlarını tamamlayın. Bu adımlar, FITS dosyalarını model için uygun formata dönüştürecektir.

## Kullanım

### 1. Model Eğitimi

Derin öğrenme modelini eğitmek için aşağıdaki komutu çalıştırın:

```bash
python -m torch.distributed.launch --nproc_per_node=8 src/train.py
```

- **`--nproc_per_node=8`**: Eğitimde kullanılacak GPU sayısını belirtir.

### 2. Model Değerlendirmesi

Eğitilmiş modeli test etmek ve değerlendirmek için aşağıdaki komutu kullanın:

```bash
python src/evaluate.py
```

### 3. Tespit ve Analiz

Kaynakları tespit etmek ve analiz etmek için Jupyter Notebook'u çalıştırın:

```bash
jupyter notebook notebooks/detect_and_analyze.ipynb
```

## Proje Yapısı

```
astro-psf-detection/
├── data/
│   ├── raw/                      # Ham FITS görüntü verileri
│   ├── processed/                # İşlenmiş veriler
│   ├── annotations/              # Etiketlenmiş veriler (koordinatlar, sınıflar vb.)
│   ├── results/                  # Model sonuçları ve çıktı dosyaları
│
├── notebooks/
│   ├── data_preprocessing.ipynb  # Veri ön işleme adımları
│   ├── model_training.ipynb      # Model eğitim süreci
│   ├── model_evaluation.ipynb    # Model değerlendirme ve analiz
│   ├── detect_and_analyze.ipynb  # Tespit ve analiz işlemleri
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Veri ön işleme ve hazırlama
│   ├── model.py                  # Derin öğrenme modeli tanımı
│   ├── train.py                  # Model eğitim scripti
│   ├── evaluate.py               # Model değerlendirme scripti
│   ├── download_data.py          # Veri indirme ve çıkarma scripti
│   ├── utils.py                  # Yardımcı fonksiyonlar
│
├── config/
│   ├── config.yaml               # Model ve veri işleme ayar dosyası
│
├── tests/
│   ├── test_data_preprocessing.py# Veri ön işleme testleri
│   ├── test_model.py             # Model testleri
│   ├── test_utils.py             # Yardımcı fonksiyon testleri
│
├── results/
│   ├── figures/                  # Görselleştirme ve grafikler
│   ├── logs/                     # Eğitim ve değerlendirme log dosyaları
│
├── requirements.txt              # Gerekli Python paketleri
├── README.md                     # Proje açıklamaları ve dokümantasyonu
├── setup.py                      # Paket kurulum dosyası
└── .gitignore                    # Git için ihmal edilecek dosyalar
```

## Katkıda Bulunma

Projeye katkıda bulunmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu forklayın.
2. Yeni bir özellik veya düzeltme üzerinde çalışın.
3. Çalışmanızı test edin ve belgelerinizi güncelleyin.
4. Değişikliklerinizi gönderin ve bir pull request oluşturun.

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasını inceleyebilirsiniz.

## İletişim

Bu proje hakkında sorularınız veya önerileriniz için [m.mustafakapici@gmail.com](mailto:m.mustafakapici@gmail.com) adresine e-posta gönderebilirsiniz.

Ayrıca projenin GitHub deposunu ziyaret ederek geri bildirim bırakabilir veya katkıda bulunabilirsiniz: [GitHub - Astro Object Detection](https://github.com/mmustafakapici/astro-object-detection).

---
