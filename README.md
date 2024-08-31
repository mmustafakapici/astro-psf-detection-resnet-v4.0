
### 1. Proje Başlığı ve Açıklama
**Başlık:** Astro PSF Detection with ResNet v4.0

**Açıklama:**
Bu proje, astronomik görüntülerdeki Nokta Yayılma Fonksiyonunu (PSF) tespit etmek için ResNet-50 modelini kullanmaktadır. Projede, SDSS ve Phosim gibi veriler üzerinde eğitim ve test yapılmış, model performansı IoU, mAP gibi metriklerle değerlendirilmiştir.

### 2. İçindekiler
- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Eğitim ve Değerlendirme](#eğitim-ve-değerlendirme)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

### 3. Kurulum
Projeyi klonladıktan sonra gerekli Python paketlerini yükleyin:

```bash
git clone https://github.com/mmustafakapici/astro-psf-detection-resnet-v4.0.git
cd astro-psf-detection-resnet-v4.0
pip install -r requirements.txt
```

### 4. Veri Seti
Bu projede kullanılan veri setleri:

- **SDSS:** SDSS verileri, FITS formatında, g, r, z bantlarında indirilmiştir.
- **Phosim:** Simüle edilmiş astronomik görüntülerden oluşan bir veri seti.
  
Veri setlerine erişmek için `astroquery` kullanılmıştır.

### 5. Eğitim ve Değerlendirme
Model, ResNet-50 mimarisi temel alınarak eğitilmiştir. Eğitim süreci aşağıdaki adımları içermektedir:

- **Model Mimarisi:** ResNet-50
- **Eğitim Ayarları:** Config dosyasından belirlenmiştir.
- **Değerlendirme Metrikleri:** IoU, mAP (30, 50, 75)
- **Optimizasyon:** Adam optimizer, Learning rate scheduler

Eğitim süreci GPU destekli olarak gerçekleştirilmiştir.

### 6. Sonuçlar
Modelin eğitim ve test performansı, IoU ve mAP metrikleri üzerinden değerlendirilmiştir. Detaylı sonuçlar ve grafikler proje dizininde bulunmaktadır.

### 7. Katkıda Bulunma
Projeye katkıda bulunmak isterseniz lütfen şu adımları takip edin:

1. Projeyi forklayın.
2. Kendi branşınızı oluşturun (`git checkout -b feature/AmazingFeature`).
3. Değişikliklerinizi ekleyin (`git add .`).
4. Değişiklikleri commit edin (`git commit -m 'Add some AmazingFeature'`).
5. Branşınıza push edin (`git push origin feature/AmazingFeature`).
6. Bir Pull Request açın.

### 8. Lisans
Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.

Bu yapıyı kullanarak README dosyanızı güncelleyebiliriz. Ayrıca, özel gereksinimlerinizi eklemek isterseniz detayları paylaşabilirsiniz.

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasını inceleyebilirsiniz.

## İletişim

Bu proje hakkında sorularınız veya önerileriniz için [m.mustafakapici@gmail.com](mailto:m.mustafakapici@gmail.com) adresine e-posta gönderebilirsiniz.

Ayrıca projenin GitHub deposunu ziyaret ederek geri bildirim bırakabilir veya katkıda bulunabilirsiniz: [GitHub - Astro PSF Detection with ResNet v4.0](https://github.com/mmustafakapici/astro-psf-detection-resnet-v4.0).

---