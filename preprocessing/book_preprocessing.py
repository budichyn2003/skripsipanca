import pandas as pd
import numpy as np
import re
import string
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Natural Language Processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results

class BookPreprocessor:
    """
    Kelas untuk preprocessing data buku dalam sistem rekomendasi content-based filtering
    Mendukung preprocessing teks campuran bahasa Indonesia dan Inggris
    """
    
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize stemmers and lemmatizers
        self.indonesian_stemmer = StemmerFactory().create_stemmer()
        self.english_lemmatizer = WordNetLemmatizer()
        
        # Initialize stopwords
        self.indonesian_stopwords = self._get_indonesian_stopwords()
        self.english_stopwords = set(stopwords.words('english'))
        self.combined_stopwords = self.indonesian_stopwords.union(self.english_stopwords)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def _get_indonesian_stopwords(self) -> set:
        """Get Indonesian stopwords using Sastrawi"""
        factory = StopWordRemoverFactory()
        stopwords_list = factory.get_stop_words()
        return set(stopwords_list)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset dari file CSV
        
        Args:
            file_path (str): Path ke file CSV
            
        Returns:
            pd.DataFrame: Dataset yang sudah dimuat
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tahap 1: Data Cleaning
        - Hapus NaN/null values pada kolom penting
        - Hapus duplikat judul
        - Gabungkan kolom relevan
        
        Args:
            df (pd.DataFrame): Dataset asli
            
        Returns:
            pd.DataFrame: Dataset yang sudah dibersihkan
        """
        print("=== TAHAP 1: DATA CLEANING ===")
        
        # Buat copy dataset
        df_clean = df.copy()
        
        # Kolom yang penting untuk content-based filtering
        important_columns = ['judul_utama', 'deskripsi', 'subjek_topik']
        
        # Cek missing values sebelum cleaning
        print("Missing values sebelum cleaning:")
        for col in important_columns:
            if col in df_clean.columns:
                missing_count = df_clean[col].isnull().sum()
                print(f"  {col}: {missing_count} missing values")
        
        # Hapus baris dengan NaN pada kolom penting
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=important_columns, how='any')
        removed_rows = initial_rows - len(df_clean)
        print(f"Dihapus {removed_rows} baris dengan missing values")
        
        # Hapus duplikat berdasarkan judul_utama
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['judul_utama'], keep='first')
        removed_duplicates = initial_rows - len(df_clean)
        print(f"Dihapus {removed_duplicates} duplikat judul")
        
        # Gabungkan kolom relevan untuk representasi yang lebih kaya
        df_clean['combined_features'] = (
            df_clean['judul_utama'].astype(str) + ' ' +
            df_clean['deskripsi'].astype(str) + ' ' +
            df_clean['subjek_topik'].astype(str)
        )
        
        # Tambahkan kategori jika ada
        if 'kategori' in df_clean.columns:
            df_clean['combined_features'] += ' ' + df_clean['kategori'].astype(str)
        
        print(f"Dataset setelah cleaning: {len(df_clean)} baris")
        print("Kolom 'combined_features' berhasil dibuat")
        
        return df_clean
    
    def case_folding(self, text: str) -> str:
        """
        Tahap 2: Case Folding - ubah ke lowercase
        
        Args:
            text (str): Teks input
            
        Returns:
            str: Teks dalam lowercase
        """
        if pd.isna(text):
            return ""
        return str(text).lower()
    
    def remove_punctuation_numbers(self, text: str) -> str:
        """
        Tahap 3: Remove Punctuation & Numbers
        
        Args:
            text (str): Teks input
            
        Returns:
            str: Teks tanpa tanda baca dan angka
        """
        if pd.isna(text):
            return ""
        
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Hapus karakter khusus lainnya
        text = re.sub(r'[^\w\s]', '', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenization(self, text: str) -> List[str]:
        """
        Tahap 4: Tokenization
        
        Args:
            text (str): Teks input
            
        Returns:
            List[str]: List token/kata
        """
        if pd.isna(text) or text == "":
            return []
        
        try:
            tokens = word_tokenize(text)
            return [token for token in tokens if token.strip()]
        except:
            # Fallback jika NLTK tokenizer gagal
            return text.split()
    
    def remove_stopwords(self, tokens: List[str], language: Optional[str] = None) -> List[str]:
        """
        Tahap 5: Stopwords Removal
        
        Args:
            tokens (List[str]): List token
            language (str, optional): Bahasa ('indonesia' atau 'inggris')
            
        Returns:
            List[str]: Token tanpa stopwords
        """
        if not tokens:
            return []
        
        if language == 'indonesia':
            stopwords_to_use = self.indonesian_stopwords
        elif language == 'inggris':
            stopwords_to_use = self.english_stopwords
        else:
            # Gunakan gabungan stopwords jika bahasa tidak diketahui
            stopwords_to_use = self.combined_stopwords
        
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_to_use]
        return filtered_tokens
    
    def detect_language(self, text: str) -> str:
        """
        Deteksi bahasa teks
        
        Args:
            text (str): Teks input
            
        Returns:
            str: 'indonesia' atau 'inggris'
        """
        if pd.isna(text) or text.strip() == "":
            return 'unknown'
        
        try:
            detected = detect(text)
            if detected == 'id':
                return 'indonesia'
            elif detected == 'en':
                return 'inggris'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def stemming_lemmatization(self, tokens: List[str], language: str) -> List[str]:
        """
        Tahap 6: Stemming/Lemmatization
        
        Args:
            tokens (List[str]): List token
            language (str): Bahasa ('indonesia' atau 'inggris')
            
        Returns:
            List[str]: Token yang sudah di-stem/lemmatize
        """
        if not tokens:
            return []
        
        processed_tokens = []
        
        for token in tokens:
            if language == 'indonesia':
                # Gunakan Sastrawi stemmer untuk bahasa Indonesia
                stemmed = self.indonesian_stemmer.stem(token)
                processed_tokens.append(stemmed)
            elif language == 'inggris':
                # Gunakan NLTK lemmatizer untuk bahasa Inggris
                lemmatized = self.english_lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
            else:
                # Jika bahasa tidak diketahui, gunakan token asli
                processed_tokens.append(token)
        
        return processed_tokens
    
    def preprocess_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Pipeline lengkap preprocessing teks
        
        Args:
            text (str): Teks input
            language (str, optional): Bahasa teks
            
        Returns:
            str: Teks yang sudah dipreprocess
        """
        # Tahap 2: Case Folding
        text = self.case_folding(text)
        
        # Tahap 3: Remove Punctuation & Numbers
        text = self.remove_punctuation_numbers(text)
        
        # Deteksi bahasa jika tidak diberikan
        if language is None:
            language = self.detect_language(text)
        
        # Tahap 4: Tokenization
        tokens = self.tokenization(text)
        
        # Tahap 5: Stopwords Removal
        tokens = self.remove_stopwords(tokens, language)
        
        # Tahap 6: Stemming/Lemmatization
        tokens = self.stemming_lemmatization(tokens, language)
        
        # Gabungkan kembali menjadi string
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess seluruh dataset
        
        Args:
            df (pd.DataFrame): Dataset yang sudah di-clean
            
        Returns:
            pd.DataFrame: Dataset yang sudah dipreprocess
        """
        print("\n=== TAHAP 2-6: TEXT PREPROCESSING ===")
        
        df_processed = df.copy()
        
        # Preprocess combined_features
        print("Memproses kolom 'combined_features'...")
        
        processed_texts = []
        for idx, row in df_processed.iterrows():
            text = row['combined_features']
            language = row.get('bahasa', None)
            
            # Mapping bahasa
            if language == 'indonesia':
                lang_code = 'indonesia'
            elif language == 'inggris':
                lang_code = 'inggris'
            else:
                lang_code = None
            
            processed_text = self.preprocess_text(text, lang_code)
            processed_texts.append(processed_text)
            
            if (idx + 1) % 100 == 0:
                print(f"  Diproses: {idx + 1}/{len(df_processed)} buku")
        
        df_processed['processed_features'] = processed_texts
        
        # Hapus baris dengan processed_features kosong
        initial_rows = len(df_processed)
        df_processed = df_processed[df_processed['processed_features'].str.strip() != '']
        removed_empty = initial_rows - len(df_processed)
        
        if removed_empty > 0:
            print(f"Dihapus {removed_empty} baris dengan teks kosong setelah preprocessing")
        
        print(f"Preprocessing selesai: {len(df_processed)} buku siap untuk vectorization")
        
        return df_processed
    
    def vectorization(self, df: pd.DataFrame, max_features: int = 5000) -> pd.DataFrame:
        """
        Tahap 7: Vectorization dengan TF-IDF
        
        Args:
            df (pd.DataFrame): Dataset yang sudah dipreprocess
            max_features (int): Maksimal fitur untuk TF-IDF
            
        Returns:
            pd.DataFrame: Dataset dengan matrix TF-IDF
        """
        print(f"\n=== TAHAP 7: VECTORIZATION (TF-IDF) ===")
        
        # Inisialisasi TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigram dan bigram
            min_df=2,  # Minimal muncul di 2 dokumen
            max_df=0.8  # Maksimal muncul di 80% dokumen
        )
        
        # Fit dan transform teks
        processed_texts = df['processed_features'].tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        print(f"TF-IDF Matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return df
    
    def run_full_preprocessing(self, file_path: str, max_features: int = 5000) -> pd.DataFrame:
        """
        Jalankan seluruh pipeline preprocessing
        
        Args:
            file_path (str): Path ke file dataset
            max_features (int): Maksimal fitur untuk TF-IDF
            
        Returns:
            pd.DataFrame: Dataset yang sudah siap untuk sistem rekomendasi
        """
        print("🔹 MEMULAI PREPROCESSING SISTEM REKOMENDASI BUKU 🔹")
        print("=" * 60)
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Data cleaning
        df_clean = self.data_cleaning(df)
        
        # Text preprocessing
        df_processed = self.preprocess_dataset(df_clean)
        
        # Vectorization
        df_final = self.vectorization(df_processed, max_features)
        
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING SELESAI!")
        print(f"Dataset final: {len(df_final)} buku")
        print(f"Fitur TF-IDF: {self.tfidf_matrix.shape[1]} fitur")
        
        return df_final
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Simpan data yang sudah dipreprocess
        
        Args:
            df (pd.DataFrame): Dataset yang sudah dipreprocess
            output_path (str): Path output file
        """
        try:
            df.to_csv(output_path, index=False)
            print(f"Data berhasil disimpan ke: {output_path}")
        except Exception as e:
            print(f"Error menyimpan data: {e}")


# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi preprocessor
    preprocessor = BookPreprocessor()
    
    # Jalankan preprocessing
    input_file = "../eaa.csv"  # Sesuaikan path
    output_file = "../data/processed_books.csv"
    
    # Preprocessing lengkap
    df_processed = preprocessor.run_full_preprocessing(input_file, max_features=5000)
    
    if df_processed is not None:
        # Simpan hasil
        preprocessor.save_processed_data(df_processed, output_file)
        
        # Tampilkan contoh hasil
        print("\n=== CONTOH HASIL PREPROCESSING ===")
        print("Original text:")
        print(df_processed.iloc[0]['combined_features'][:200] + "...")
        print("\nProcessed text:")
        print(df_processed.iloc[0]['processed_features'][:200] + "...")
