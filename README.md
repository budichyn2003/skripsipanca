# 📚 Sistem Rekomendasi Buku Content-Based Filtering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org)

Sistem rekomendasi buku berbasis Content-Based Filtering menggunakan TF-IDF dan Cosine Similarity untuk memberikan rekomendasi buku yang relevan berdasarkan konten dan karakteristik buku.

## 🎯 Tujuan Proyek

Membangun sistem rekomendasi yang dapat:
- Memberikan rekomendasi buku berdasarkan judul buku yang dipilih
- Mencari buku berdasarkan keyword tertentu
- Memberikan rekomendasi berdasarkan kriteria (kategori, bahasa, subjek)
- Menganalisis similarity dan karakteristik dataset buku

## 🏗️ Arsitektur Sistem

```
📁 skripsipanca/
├── 📁 data/
│   ├── processed_books.csv          # Dataset buku yang sudah dipreprocess
│   └── ea.xlsx                      # Dataset asli
├── 📁 preprocessing/
│   └── book_preprocessing.py        # Modul preprocessing data
├── 📁 model/
│   ├── content_based_recommender.py # Sistem rekomendasi utama
│   ├── book_recommender_model.pkl   # Model terlatih (pickle)
│   └── book_recommender_model_summary.json # Summary model (JSON)
├── 📁 cbf_venv/                     # Virtual environment
├── similarity_analysis.py           # Analisis similarity matrix
├── recommendation_demo.py           # Demo interaktif sistem
├── run_preprocessing.py             # Script menjalankan preprocessing
├── run_similarity_analysis.py       # Script analisis similarity
└── requirements.txt                 # Dependencies
```

## 🔄 Pipeline Sistem

### 1. **Data Preprocessing**
```python
# Jalankan preprocessing
python run_preprocessing.py
```

**Proses yang dilakukan:**
- **Text Cleaning**: Menghapus karakter khusus, angka, dan tanda baca
- **Lowercasing**: Mengubah semua teks menjadi huruf kecil
- **Stopword Removal**: Menghapus kata-kata umum (stopwords)
- **Feature Engineering**: Menggabungkan judul, deskripsi, kategori, dan subjek
- **Normalization**: Normalisasi teks untuk konsistensi

**Input:** `ea.xlsx` (dataset asli)  
**Output:** `data/processed_books.csv` (dataset bersih)

### 2. **Model Building & Training**
```python
# Membangun sistem rekomendasi
from model.content_based_recommender import ContentBasedRecommender

recommender = ContentBasedRecommender()
recommender.build_recommendation_system("data/processed_books.csv")
```

**Proses yang dilakukan:**
- **TF-IDF Vectorization**: Mengubah teks menjadi vektor numerik
- **Similarity Matrix Calculation**: Menghitung cosine similarity antar buku
- **Model Serialization**: Menyimpan model dalam format pickle dan JSON

### 3. **Similarity Analysis**
```python
# Jalankan analisis similarity
python run_similarity_analysis.py
```

**Analisis yang dilakukan:**
- **Coverage Analysis**: Berapa buku yang bisa direkomendasikan
- **Threshold Impact Analysis**: Trade-off precision vs coverage
- **Degree Distribution**: Analisis konektivitas antar buku
- **Sparsity Analysis**: Efisiensi representasi data

### 4. **Recommendation System**
```python
# Demo sistem rekomendasi
python recommendation_demo.py
```

**Fitur yang tersedia:**
- Rekomendasi berdasarkan judul buku
- Pencarian berdasarkan keyword
- Rekomendasi berdasarkan kriteria (kategori, bahasa, subjek)
- Mode interaktif untuk user

## 🧠 Metodologi

### **Content-Based Filtering**
Sistem menggunakan pendekatan Content-Based Filtering yang merekomendasikan item berdasarkan kesamaan konten dengan item yang disukai user.

### **TF-IDF (Term Frequency-Inverse Document Frequency)**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

dimana:
- TF(t,d) = frekuensi term t dalam dokumen d
- IDF(t) = log(N / df(t))
- N = total dokumen
- df(t) = jumlah dokumen yang mengandung term t
```

**Keunggulan TF-IDF:**
- Memberikan bobot tinggi pada kata yang sering muncul dalam dokumen tertentu
- Memberikan bobot rendah pada kata yang umum di seluruh koleksi
- Efektif untuk representasi teks dalam sistem rekomendasi

### **Cosine Similarity**
```
similarity(A,B) = cos(θ) = (A·B) / (||A|| × ||B||)

dimana:
- A, B = vektor TF-IDF dari dua buku
- A·B = dot product dari vektor A dan B
- ||A||, ||B|| = magnitude (panjang) vektor A dan B
```

**Rentang nilai:** 0 (tidak mirip) hingga 1 (identik)

## 📊 Hasil Evaluasi

### **Dataset Overview**
- **Total Buku**: 97 buku
- **Kategori**: Accounting/Akuntansi
- **Bahasa**: Indonesia (47), Inggris (49), Indonesia (1)
- **TF-IDF Features**: 1,146 fitur unik
- **Sparsity**: 98.2% (efisien secara memori)

### **Similarity Statistics**
```json
{
  "mean_similarity": 0.028514,
  "max_similarity": 0.670348,
  "percentiles": {
    "p95": 0.119966,
    "p99": 0.26075
  }
}
```

### **Coverage Analysis**
| Threshold | Coverage | Avg Recommendations |
|-----------|----------|-------------------|
| 0.05      | 100.0%   | 16.2 buku/buku   |
| 0.10      | 94.85%   | 6.3 buku/buku    |
| 0.15      | 82.47%   | 3.3 buku/buku    |
| 0.20      | 67.01%   | 1.8 buku/buku    |

**Rekomendasi Optimal**: Threshold 0.1 (94.85% coverage, 6.3 rekomendasi per buku)

### **Top TF-IDF Features**
1. **accounting** (0.055745)
2. **akuntansi accountingakuntansi** (0.048598)
3. **buku** (0.044091)
4. **dasar** (0.034889)
5. **teori** (0.029905)

## 🚀 Instalasi & Penggunaan

### **1. Clone Repository**
```bash
git clone https://github.com/username/skripsipanca.git
cd skripsipanca
```

### **2. Setup Virtual Environment**
```bash
python -m venv cbf_venv
# Windows
cbf_venv\Scripts\activate
# Linux/Mac
source cbf_venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Jalankan Preprocessing (Opsional)**
```bash
python run_preprocessing.py
```

### **5. Jalankan Sistem Rekomendasi**
```bash
python recommendation_demo.py
```

### **6. Analisis Similarity (Opsional)**
```bash
python run_similarity_analysis.py
```

## 💻 Contoh Penggunaan

### **Rekomendasi Berdasarkan Judul**
```python
from model.content_based_recommender import ContentBasedRecommender

recommender = ContentBasedRecommender()
recommender.build_recommendation_system("data/processed_books.csv")

# Dapatkan rekomendasi
recommendations = recommender.get_book_recommendations("Accounting", top_n=5)

for rec in recommendations:
    print(f"{rec['rank']}. {rec['judul']} (Score: {rec['similarity_score']})")
```

### **Pencarian Berdasarkan Keyword**
```python
# Cari buku dengan keyword
search_results = recommender.search_books_by_keyword("akuntansi", top_n=5)

for book in search_results:
    print(f"- {book['judul']}")
    print(f"  Deskripsi: {book['deskripsi'][:100]}...")
```

### **Rekomendasi Berdasarkan Kriteria**
```python
# Filter berdasarkan kriteria
criteria = {
    'kategori': 'Accounting',
    'bahasa': 'indonesia',
    'subjek_topik': 'akuntansi'
}

recommendations = recommender.get_recommendations_by_features(criteria, top_n=5)
```

## 📈 Hasil Rekomendasi

### **Contoh Output Sistem**
```
🔍 Mencari rekomendasi untuk buku: 'Accounting'
✅ Ditemukan 5 rekomendasi:

1. 📚 Fundamental accounting principles
   👤 Pengarang: Larson, Kermit D.
   🏢 Penerbit: Richard D. Irwin, (1996.0)
   📂 Kategori: Accounting/Akuntansi
   🌐 Bahasa: inggris
   ⭐ Similarity Score: 0.1972

2. 📚 Accounting
   👤 Pengarang: Horngren, Charles T.
   🏢 Penerbit: Prentice Hall, (2001.0)
   📂 Kategori: Accounting/Akuntansi
   🌐 Bahasa: inggris
   ⭐ Similarity Score: 0.1427
```

## 🔧 Konfigurasi

### **TF-IDF Parameters**
```python
TfidfVectorizer(
    max_features=5000,      # Maksimal fitur
    ngram_range=(1, 2),     # Unigram dan bigram
    min_df=1,               # Minimal muncul di 1 dokumen
    max_df=0.8,             # Maksimal muncul di 80% dokumen
    stop_words=None         # Sudah dihapus di preprocessing
)
```

### **Similarity Threshold**
- **Rendah (0.05)**: Lebih banyak rekomendasi, precision rendah
- **Sedang (0.10)**: Seimbang antara coverage dan precision
- **Tinggi (0.20)**: Sedikit rekomendasi, precision tinggi

## 📁 File Output

### **Model Files**
- `model/book_recommender_model.pkl`: Model terlatih (untuk loading)
- `model/book_recommender_model_summary.json`: Summary model (untuk inspeksi)

### **Analysis Files**
- `similarity_analysis_[timestamp].json`: Hasil analisis similarity
- `similarity_analysis_report_[timestamp].html`: Laporan HTML
- `similarity_matrix_[timestamp].csv`: Similarity matrix
- `similarity_matrix_heatmap.png`: Visualisasi heatmap

## 🛠️ Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

## 📊 Evaluasi & Metrics

### **Metrics yang Digunakan**
1. **Coverage**: Persentase buku yang bisa direkomendasikan
2. **Precision@K**: Akurasi rekomendasi top-K
3. **Similarity Distribution**: Distribusi nilai similarity
4. **Sparsity**: Efisiensi representasi data

### **Kelebihan Sistem**
✅ **Tidak memerlukan data user**: Content-based approach  
✅ **Transparan**: Rekomendasi berdasarkan kesamaan konten yang jelas  
✅ **Scalable**: Efisien untuk dataset besar  
✅ **Cold Start**: Dapat merekomendasikan item baru  

### **Keterbatasan Sistem**
❌ **Limited Diversity**: Cenderung merekomendasikan item serupa  
❌ **Content Dependency**: Bergantung pada kualitas deskripsi konten  
❌ **No User Personalization**: Tidak mempelajari preferensi individual  

## 🔮 Pengembangan Selanjutnya

1. **Hybrid Approach**: Kombinasi dengan Collaborative Filtering
2. **Deep Learning**: Implementasi neural embeddings
3. **User Profiling**: Sistem personalisasi berdasarkan riwayat user
4. **Real-time Updates**: Sistem yang dapat update secara real-time
5. **Multi-language Support**: Dukungan untuk berbagai bahasa

## 📚 Referensi

1. Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*
3. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*
4. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*

## 👨‍💻 Author

**Nama**: [Nama Anda]  
**NIM**: [NIM Anda]  
**Program Studi**: [Program Studi Anda]  
**Universitas**: [Universitas Anda]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**📧 Contact**: [email@domain.com]  
**🔗 GitHub**: [https://github.com/username/skripsipanca](https://github.com/username/skripsipanca)
