# U-Net Aerial Imagery Segmentation — Tasarım Kararları

## Bağlam
Kaynak: kurs videosundaki basit `load_dataset` fonksiyonu (cv2 ile resize+normalize).
Hedef: aynı veri seti (Kaggle "Semantic segmentation of aerial imagery") için
daha temiz, modüler ve doğru bir pipeline kurmak.

## Kararlar ve gerekçeleri

| Konu | Karar | Gerekçe |
|---|---|---|
| Framework | PyTorch | Python 3.14 kurulu, TensorFlow bu sürümü desteklemiyor (pip'te wheel yok); torch zaten çalışıyor. |
| Kapsam | Tam pipeline | Data + model + train + eval + predict, gerçek bir proje gibi çalışsın. |
| Ortam | Yerel Mac, CPU | GPU yok; batch size ve epoch buna göre config'den ayarlanabilir tutuldu. |
| Görüntü boyutu | 256x256 patchify | Tile'lar arası boyut 509x544–2149x1479 arası değişken; resize aşırı distorsiyona yol açar. Patch'lere bölmek hem boyutu sabitler hem veri miktarını artırır. |
| Mask formatı | RGB -> class-index dönüşümü | Maskeler grayscale değil, renk kodlu. classes.json'daki hex renkler gerçek piksellerle uyuşmuyor; gerçek renkler ölçülüp `mask_utils.py`'de sabitlendi. |
| Split stratejisi | Tile bazında (6 train / 1 val / 1 test) | Aynı görüntüden çıkan patch'lerin train ve test'e dağılıp veri sızıntısına yol açmasını önlemek için. |
| Loss | Class-weighted CrossEntropyLoss | Sınıf dağılımı dengesiz (Unlabeled/Vegetation baskın, Water az); ağırlıklandırma train patch cache'inden otomatik hesaplanıyor. |

## Doğrulanan veri gerçekleri
- 8 tile, toplam 72 görüntü, 9 görüntü/tile.
- Görüntü boyutları: Tile1 797x644, Tile2 509x544, Tile3 682x658, Tile4 1099x846,
  Tile5 1126x1058, Tile6 859x838, Tile7 1817x2061, Tile8 2149x1479.
- Mask piksel renkleri (ölçülen): Building #3C1098, Land #8429F6, Road #6EC1E4,
  Vegetation #FEDD3A, Water #E2A929, Unlabeled #9B9B9B.

## Uygulanmadı / sonraki adım olabilir
- Albumentations tabanlı zengin augmentation (şimdilik flip/rotate90, torch/numpy ile, ek bağımlılık istenmedi).
- Git repository henüz kurulmadı (proje dizini `git init` içermiyor).
- GPU/Colab'a taşıma script'leri (device config'den `cpu`/`cuda`/`mps` olarak değiştirilebilir, kod platform-agnostik yazıldı).
