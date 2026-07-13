# ABSA IndoBERT — Sentimen Pascabencana Aceh

Sistem *Aspect-Based Sentiment Analysis* (ABSA) berbasis IndoBERT untuk menganalisis opini publik terkait penanganan pascabencana hidrometeorologi di Aceh.

## Fitur

- Klasifikasi sentimen otomatis (negatif, netral, positif)
- Analisis 5 aspek penanganan bencana secara paralel
- Antarmuka web interaktif dengan visualisasi Chart.js
- Model *fine-tuned* IndoBERT pada dataset opini Twitter

## Aspek yang Dianalisis

| Aspek | Contoh Kata Kunci |
|-------|-------------------|
| Logistik dan Bantuan | sembako, makanan, tenda, donasi, bantuan |
| Evakuasi dan Pengungsian | evakuasi, pengungsi, relokasi, basarnas |
| Koordinasi Pemerintah | pemerintah, bpbd, bnpb, gubernur, kebijakan |
| Infrastruktur dan Rekonstruksi | jalan, jembatan, listrik, rekonstruksi |
| Layanan Kesehatan | kesehatan, dokter, obat, puskesmas |

## Arsitektur

```
Input Teks → Pra-pemrosesan (indoNLP) → Deteksi Aspek → Tokenisasi IndoBERT → Klasifikasi → Output JSON
```

## Teknologi

- **Model:** IndoBERT (`indobenchmark/indobert-base-p1`)
- **Backend:** Python 3.10, Flask, Waitress
- **Frontend:** Chart.js, Tailwind CSS
- **Deploy:** Render.com

## Evaluasi Model

| Metrik | Nilai |
|--------|-------|
| Accuracy | 91.27% |
| Precision (macro) | 86.83% |
| Recall (macro) | 85.37% |
| F1-Score (macro) | 86.03% |

## Instalasi Lokal

```bash
# Clone repository
git clone https://github.com/l0r3ntz/absa-deploy.git
cd absa-deploy

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
python app.py
```

Buka `http://localhost:5000` di browser.

## API

### `POST /predict`

```json
// Request
{ "text": "Bantuan logistik sangat lambat sampai ke korban banjir" }

// Response
{
  "input_text": "Bantuan logistik sangat lambat sampai ke korban banjir",
  "results": [
    {
      "aspect_slug": "logistik_bantuan",
      "aspect_name": "Logistik dan bantuan",
      "sentimen_prediksi": "negatif",
      "confidence": 0.94
    }
  ],
  "total_aspects": 1
}
```

### `GET /health`

```json
{ "status": "ok" }
```

## Struktur Project

```
absa-deploy/
├── app.py                    # Flask server
├── absa_predictor.py         # Kelas prediksi ABSA
├── requirements.txt          # Dependencies
├── render.yaml               # Render config
├── templates/
│   └── index.html            # Antarmuka web
└── model_indobert_absa/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Lisensi

Project ini dibuat untuk keperluan penelitian skripsi.
