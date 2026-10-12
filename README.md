# Klasifikasi Penyakit Daun Tomat (Deep Learning)

Proyek klasifikasi citra untuk mengenali tiga kondisi daun tomat:
**healthy**, **late_blight**, dan **septoria_leaf_spot**, menggunakan CNN
(TensorFlow/Keras), dengan prototipe Streamlit.

- Demo aplikasi: [Tomato Disease Classification](https://tomato-disease-classification-vkcnc2zgndcdtb4a4rtech.streamlit.app/)
- Notebook: `Tomato_Disease_By_Leaf_Image_Classification.ipynb`
- Dataset: [cookiefinder/tomato-disease-multiple-sources](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources)

## Hasil (model v1, data uji 1634 gambar)

| Metrik    | Nilai  |
| --------- | ------ |
| Akurasi   | 92,96% |
| Macro F1  | 0,930  |
| Macro AUC | 0,990  |

## Struktur

```
app.py                      # prototipe Streamlit
requirements.txt
models/
  v1/
    tomato_disease_model.tflite
    metadata.json           # versi, arsitektur, metrik, pembagian data
Tomato_Disease_By_Leaf_Image_Classification.ipynb
```

## Menjalankan secara lokal

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Versioning model

Setiap versi model disimpan di `models/vN/` bersama `metadata.json`.
Untuk versi baru, tambahkan `models/v2/` lalu ubah `MODEL_DIR` di `app.py`.

## Keterbatasan

- Dilatih pada foto daun tunggal dengan latar relatif seragam.
- Terdapat gejala _overfitting_.
- Alat bantu awal, bukan pengganti ahli pertanian.
