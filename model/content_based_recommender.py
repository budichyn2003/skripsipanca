import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Tuple

class ContentBasedRecommender:
    """
    Sistem Rekomendasi Buku menggunakan Content-Based Filtering
    Menggunakan TF-IDF dan Cosine Similarity untuk mencari buku serupa
    """
    
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.book_indices = None
        
    def load_processed_data(self, file_path: str) -> bool:
        """
        Load data yang sudah dipreprocess
        
        Args:
            file_path (str): Path ke file processed_books.csv
            
        Returns:
            bool: True jika berhasil load data
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Data berhasil dimuat: {len(self.df)} buku")
            
            # Buat mapping index untuk pencarian cepat
            self.book_indices = pd.Series(self.df.index, index=self.df['judul_utama'].str.lower().str.strip()).drop_duplicates()
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def build_tfidf_matrix(self, max_features: int = 5000) -> bool:
        """
        Membangun TF-IDF matrix dari processed features
        
        Args:
            max_features (int): Maksimal fitur TF-IDF
            
        Returns:
            bool: True jika berhasil build matrix
        """
        try:
            print("🔄 Membangun TF-IDF Matrix...")
            
            # Inisialisasi TF-IDF Vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Unigram dan bigram
                min_df=1,  # Minimal muncul di 1 dokumen
                max_df=0.8,  # Maksimal muncul di 80% dokumen
                stop_words=None  # Sudah dihapus di preprocessing
            )
            
            # Fit dan transform processed features
            processed_texts = self.df['processed_features'].fillna('').tolist()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            
            print(f"✅ TF-IDF Matrix berhasil dibuat:")
            print(f"   Shape: {self.tfidf_matrix.shape}")
            print(f"   Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
            
            return True
        except Exception as e:
            print(f"❌ Error building TF-IDF matrix: {e}")
            return False
    
    def calculate_similarity_matrix(self) -> bool:
        """
        Menghitung similarity matrix menggunakan cosine similarity
        
        Returns:
            bool: True jika berhasil calculate similarity
        """
        try:
            print("🔄 Menghitung Cosine Similarity Matrix...")
            
            # Hitung cosine similarity
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
            print(f"✅ Similarity Matrix berhasil dihitung:")
            print(f"   Shape: {self.similarity_matrix.shape}")
            
            return True
        except Exception as e:
            print(f"❌ Error calculating similarity matrix: {e}")
            return False
    
    def get_book_recommendations(self, book_title: str, top_n: int = 10) -> List[Dict]:
        """
        Mendapatkan rekomendasi buku berdasarkan judul buku
        
        Args:
            book_title (str): Judul buku yang dijadikan referensi
            top_n (int): Jumlah rekomendasi yang diinginkan
            
        Returns:
            List[Dict]: List rekomendasi buku dengan similarity score
        """
        try:
            # Cari index buku berdasarkan judul
            book_title_lower = book_title.lower().strip()
            
            if book_title_lower not in self.book_indices:
                # Coba pencarian fuzzy
                possible_matches = []
                for title in self.book_indices.index:
                    if book_title_lower in title or title in book_title_lower:
                        possible_matches.append(title)
                
                if possible_matches:
                    print(f"📚 Buku '{book_title}' tidak ditemukan. Mungkin maksud Anda:")
                    for i, match in enumerate(possible_matches[:5], 1):
                        original_title = self.df.iloc[self.book_indices[match]]['judul_utama']
                        print(f"   {i}. {original_title}")
                    return []
                else:
                    print(f"❌ Buku '{book_title}' tidak ditemukan dalam database")
                    return []
            
            # Dapatkan index buku
            book_idx = self.book_indices[book_title_lower]
            
            # Handle jika ada multiple matches (ambil yang pertama)
            if hasattr(book_idx, '__len__') and len(book_idx) > 1:
                book_idx = book_idx.iloc[0]
            
            # Dapatkan similarity scores untuk buku ini
            sim_scores = list(enumerate(self.similarity_matrix[book_idx]))
            
            # Sort berdasarkan similarity score (descending)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Ambil top_n+1 (karena yang pertama adalah buku itu sendiri)
            sim_scores = sim_scores[1:top_n+1]
            
            # Dapatkan indices buku yang direkomendasikan
            book_indices_rec = [i[0] for i in sim_scores]
            
            # Buat list rekomendasi
            recommendations = []
            for i, idx in enumerate(book_indices_rec):
                book_info = {
                    'rank': i + 1,
                    'judul': self.df.iloc[idx]['judul_utama'],
                    'deskripsi': self.df.iloc[idx]['deskripsi'],
                    'pengarang': self.df.iloc[idx]['tajuk_pengarang'],
                    'penerbit': self.df.iloc[idx]['penerbit'],
                    'tahun': self.df.iloc[idx]['tahun_terbit'],
                    'kategori': self.df.iloc[idx]['kategori'],
                    'bahasa': self.df.iloc[idx]['bahasa'],
                    'subjek_topik': self.df.iloc[idx]['subjek_topik'],
                    'similarity_score': round(sim_scores[i][1], 4)
                }
                recommendations.append(book_info)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
            return []
    
    def search_books_by_keyword(self, keyword: str, top_n: int = 10) -> List[Dict]:
        """
        Mencari buku berdasarkan keyword dan memberikan rekomendasi
        
        Args:
            keyword (str): Keyword pencarian
            top_n (int): Jumlah hasil pencarian
            
        Returns:
            List[Dict]: List buku yang cocok dengan keyword
        """
        try:
            keyword_lower = keyword.lower()
            
            # Cari buku yang mengandung keyword di judul, deskripsi, atau subjek
            mask = (
                self.df['judul_utama'].str.lower().str.contains(keyword_lower, na=False) |
                self.df['deskripsi'].str.lower().str.contains(keyword_lower, na=False) |
                self.df['subjek_topik'].str.lower().str.contains(keyword_lower, na=False) |
                self.df['kategori'].str.lower().str.contains(keyword_lower, na=False)
            )
            
            matching_books = self.df[mask].head(top_n)
            
            if len(matching_books) == 0:
                print(f"❌ Tidak ditemukan buku dengan keyword '{keyword}'")
                return []
            
            # Format hasil pencarian
            search_results = []
            for idx, row in matching_books.iterrows():
                book_info = {
                    'judul': row['judul_utama'],
                    'deskripsi': row['deskripsi'],
                    'pengarang': row['tajuk_pengarang'],
                    'penerbit': row['penerbit'],
                    'tahun': row['tahun_terbit'],
                    'kategori': row['kategori'],
                    'bahasa': row['bahasa'],
                    'subjek_topik': row['subjek_topik']
                }
                search_results.append(book_info)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error searching books: {e}")
            return []
    
    def get_recommendations_by_features(self, features: Dict, top_n: int = 10) -> List[Dict]:
        """
        Mendapatkan rekomendasi berdasarkan fitur yang diinginkan user
        
        Args:
            features (Dict): Dictionary fitur (kategori, bahasa, subjek_topik)
            top_n (int): Jumlah rekomendasi
            
        Returns:
            List[Dict]: List rekomendasi buku
        """
        try:
            # Filter buku berdasarkan fitur yang diminta
            filtered_df = self.df.copy()
            
            # Debug: print available values
            print(f"🔍 Debug - Mencari dengan kriteria: {features}")
            
            if 'kategori' in features and features['kategori']:
                kategori_filter = features['kategori'].lower()
                mask = filtered_df['kategori'].str.lower().str.contains(kategori_filter, na=False)
                print(f"📊 Kategori '{kategori_filter}' ditemukan: {mask.sum()} buku")
                filtered_df = filtered_df[mask]
            
            if 'bahasa' in features and features['bahasa']:
                bahasa_filter = features['bahasa'].lower().strip()
                mask = filtered_df['bahasa'].str.lower().str.strip() == bahasa_filter
                print(f"📊 Bahasa '{bahasa_filter}' ditemukan: {mask.sum()} buku")
                filtered_df = filtered_df[mask]
            
            if 'subjek_topik' in features and features['subjek_topik']:
                subjek_filter = features['subjek_topik'].lower()
                mask = filtered_df['subjek_topik'].str.lower().str.contains(subjek_filter, na=False)
                print(f"📊 Subjek '{subjek_filter}' ditemukan: {mask.sum()} buku")
                filtered_df = filtered_df[mask]
            
            print(f"📊 Total buku setelah filter: {len(filtered_df)}")
            
            if len(filtered_df) == 0:
                print("❌ Tidak ditemukan buku dengan kriteria yang diminta")
                # Show available options
                print("\n💡 Saran kriteria yang tersedia:")
                if 'kategori' in features:
                    unique_categories = self.df['kategori'].unique()
                    print(f"   Kategori: {list(unique_categories)}")
                if 'bahasa' in features:
                    unique_languages = self.df['bahasa'].str.lower().str.strip().unique()
                    print(f"   Bahasa: {list(unique_languages)}")
                if 'subjek_topik' in features:
                    unique_subjects = self.df['subjek_topik'].str.lower().unique()[:10]
                    print(f"   Subjek (contoh): {list(unique_subjects)}")
                return []
            
            # Ambil sample random atau berdasarkan popularitas
            recommendations = filtered_df.sample(min(top_n, len(filtered_df))).to_dict('records')
            
            # Format hasil
            formatted_recommendations = []
            for i, book in enumerate(recommendations):
                book_info = {
                    'rank': i + 1,
                    'judul': book['judul_utama'],
                    'deskripsi': book['deskripsi'],
                    'pengarang': book['tajuk_pengarang'],
                    'penerbit': book['penerbit'],
                    'tahun': book['tahun_terbit'],
                    'kategori': book['kategori'],
                    'bahasa': book['bahasa'],
                    'subjek_topik': book['subjek_topik']
                }
                formatted_recommendations.append(book_info)
            
            return formatted_recommendations
            
        except Exception as e:
            print(f"❌ Error getting feature-based recommendations: {e}")
            return []
    
    def save_model(self, model_path: str) -> bool:
        """
        Simpan model yang sudah dilatih
        
        Args:
            model_path (str): Path untuk menyimpan model
            
        Returns:
            bool: True jika berhasil menyimpan
        """
        try:
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'similarity_matrix': self.similarity_matrix,
                'book_indices': self.book_indices,
                'df': self.df
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model berhasil disimpan ke: {model_path}")
            
            # Simpan juga dalam format JSON untuk inspeksi
            json_path = model_path.replace('.pkl', '_summary.json')
            self.save_model_summary_json(json_path)
            
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def save_model_summary_json(self, json_path: str) -> bool:
        """
        Simpan ringkasan model dalam format JSON untuk inspeksi
        
        Args:
            json_path (str): Path untuk menyimpan JSON summary
            
        Returns:
            bool: True jika berhasil menyimpan
        """
        try:
            import json
            import datetime
            
            # Buat summary yang bisa di-serialize ke JSON
            model_summary = {
                'metadata': {
                    'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'total_books': len(self.df),
                    'tfidf_features': self.tfidf_matrix.shape[1],
                    'similarity_matrix_shape': list(self.similarity_matrix.shape)
                },
                'tfidf_analysis': {
                    'matrix_shape': list(self.tfidf_matrix.shape),
                    'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
                    'sparsity_percentage': round((1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100, 2),
                    'top_features': self._get_top_tfidf_features(15)
                },
                'similarity_statistics': self._get_similarity_statistics(),
                'book_index_mapping': {
                    'total_unique_titles': len(self.book_indices),
                    'sample_mappings': dict(list(self.book_indices.items())[:10])
                },
                'dataset_overview': {
                    'categories': list(self.df['kategori'].value_counts().to_dict().items())[:10],
                    'languages': list(self.df['bahasa'].value_counts().to_dict().items()),
                    'top_subjects': list(self.df['subjek_topik'].value_counts().to_dict().items())[:10]
                },
                'sample_recommendations': self._get_sample_recommendations()
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(model_summary, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Model summary berhasil disimpan ke: {json_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model summary: {e}")
            return False
    
    def _get_top_tfidf_features(self, top_n: int = 15) -> list:
        """Dapatkan top TF-IDF features"""
        try:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            mean_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
            
            top_indices = mean_scores.argsort()[-top_n:][::-1]
            
            top_features = []
            for i, idx in enumerate(top_indices):
                top_features.append({
                    'rank': i + 1,
                    'feature': feature_names[idx],
                    'avg_score': round(mean_scores[idx], 6)
                })
            
            return top_features
        except:
            return []
    
    def _get_similarity_statistics(self) -> dict:
        """Dapatkan statistik similarity matrix"""
        try:
            # Ambil nilai off-diagonal (bukan diagonal utama)
            mask = ~np.eye(self.similarity_matrix.shape[0], dtype=bool)
            off_diagonal = self.similarity_matrix[mask]
            
            return {
                'mean_off_diagonal': round(float(np.mean(off_diagonal)), 6),
                'median_off_diagonal': round(float(np.median(off_diagonal)), 6),
                'std_off_diagonal': round(float(np.std(off_diagonal)), 6),
                'min_off_diagonal': round(float(np.min(off_diagonal)), 6),
                'max_off_diagonal': round(float(np.max(off_diagonal)), 6),
                'percentiles': {
                    'p25': round(float(np.percentile(off_diagonal, 25)), 6),
                    'p50': round(float(np.percentile(off_diagonal, 50)), 6),
                    'p75': round(float(np.percentile(off_diagonal, 75)), 6),
                    'p90': round(float(np.percentile(off_diagonal, 90)), 6),
                    'p95': round(float(np.percentile(off_diagonal, 95)), 6),
                    'p99': round(float(np.percentile(off_diagonal, 99)), 6)
                }
            }
        except:
            return {}
    
    def _get_sample_recommendations(self) -> dict:
        """Dapatkan contoh rekomendasi untuk beberapa buku"""
        try:
            sample_books = ['Accounting', 'Advanced accounting', 'Riset akuntansi']
            sample_recs = {}
            
            for book_title in sample_books:
                recs = self.get_book_recommendations(book_title, top_n=3)
                if recs:
                    sample_recs[book_title] = [
                        {
                            'judul': rec['judul'],
                            'similarity_score': rec['similarity_score'],
                            'kategori': rec['kategori']
                        } for rec in recs
                    ]
            
            return sample_recs
        except:
            return {}
    
    def load_model(self, model_path: str) -> bool:
        """
        Load model yang sudah dilatih
        
        Args:
            model_path (str): Path model yang akan diload
            
        Returns:
            bool: True jika berhasil load
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.book_indices = model_data['book_indices']
            self.df = model_data['df']
            
            print(f"✅ Model berhasil dimuat dari: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def build_recommendation_system(self, processed_data_path: str, max_features: int = 5000) -> bool:
        """
        Build complete recommendation system
        
        Args:
            processed_data_path (str): Path ke processed data
            max_features (int): Maksimal fitur TF-IDF
            
        Returns:
            bool: True jika berhasil build system
        """
        print("🚀 MEMBANGUN SISTEM REKOMENDASI CONTENT-BASED FILTERING")
        print("=" * 60)
        
        # Step 1: Load processed data
        if not self.load_processed_data(processed_data_path):
            return False
        
        # Step 2: Build TF-IDF matrix
        if not self.build_tfidf_matrix(max_features):
            return False
        
        # Step 3: Calculate similarity matrix
        if not self.calculate_similarity_matrix():
            return False
        
        print("\n" + "=" * 60)
        print("✅ SISTEM REKOMENDASI BERHASIL DIBANGUN!")
        print(f"📊 Total buku dalam sistem: {len(self.df)}")
        print(f"🔍 Fitur TF-IDF: {self.tfidf_matrix.shape[1]}")
        print("🎯 Sistem siap memberikan rekomendasi!")
        
        return True


# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi recommender
    recommender = ContentBasedRecommender()
    
    # Build sistem rekomendasi
    processed_data_path = "../data/processed_books.csv"
    
    if recommender.build_recommendation_system(processed_data_path):
        # Simpan model
        model_path = "book_recommender_model.pkl"
        recommender.save_model(model_path)
        
        print("\n🎯 CONTOH PENGGUNAAN SISTEM REKOMENDASI:")
        print("-" * 50)
        
        # Contoh 1: Rekomendasi berdasarkan judul buku
        print("\n1. Rekomendasi berdasarkan buku 'Accounting':")
        recommendations = recommender.get_book_recommendations("Accounting", top_n=5)
        
        for rec in recommendations:
            print(f"   {rec['rank']}. {rec['judul']} (Score: {rec['similarity_score']})")
            print(f"      Pengarang: {rec['pengarang']}")
            print(f"      Kategori: {rec['kategori']}")
            print()
        
        # Contoh 2: Pencarian berdasarkan keyword
        print("\n2. Pencarian buku dengan keyword 'akuntansi':")
        search_results = recommender.search_books_by_keyword("akuntansi", top_n=3)
        
        for i, book in enumerate(search_results, 1):
            print(f"   {i}. {book['judul']}")
            print(f"      Deskripsi: {book['deskripsi'][:100]}...")
            print()
