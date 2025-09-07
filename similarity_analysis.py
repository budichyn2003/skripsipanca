"""
Script untuk Analisis Detail Similarity Calculation dalam Content-Based Filtering
Menunjukkan proses step-by-step dari TF-IDF hingga Cosine Similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class SimilarityAnalyzer:
    """
    Kelas untuk menganalisis proses similarity calculation secara detail
    """
    
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.feature_names = None
        self.analysis_results = {}  # Store all analysis results
        
    def load_processed_data(self, file_path: str) -> bool:
        """Load data yang sudah dipreprocess"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Data berhasil dimuat: {len(self.df)} buku")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_tfidf_process(self, max_features: int = 1000, show_top_features: int = 20):
        """
        Analisis proses TF-IDF secara detail
        
        Args:
            max_features (int): Maksimal fitur TF-IDF
            show_top_features (int): Jumlah fitur teratas yang ditampilkan
        """
        print("\n" + "="*70)
        print("🔍 ANALISIS PROSES TF-IDF VECTORIZATION")
        print("="*70)
        
        # Inisialisasi TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            stop_words=None
        )
        
        # Ambil processed features
        processed_texts = self.df['processed_features'].fillna('').tolist()
        
        print(f"📊 Dataset Info:")
        print(f"   - Total dokumen: {len(processed_texts)}")
        print(f"   - Max features: {max_features}")
        
        # Fit dan transform
        print(f"\n🔄 Melakukan TF-IDF transformation...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"✅ TF-IDF Matrix berhasil dibuat:")
        print(f"   - Shape: {self.tfidf_matrix.shape}")
        print(f"   - Vocabulary size: {len(self.feature_names)}")
        print(f"   - Sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100:.2f}%")
        
        # Analisis fitur dengan TF-IDF score tertinggi
        print(f"\n📈 Top {show_top_features} fitur dengan TF-IDF score tertinggi:")
        
        # Hitung rata-rata TF-IDF score per fitur
        mean_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-show_top_features:][::-1]
        
        top_features_data = []
        for i, idx in enumerate(top_indices, 1):
            feature = self.feature_names[idx]
            score = mean_scores[idx]
            print(f"   {i:2d}. {feature:<20} (avg score: {score:.4f})")
            top_features_data.append({
                'rank': i,
                'feature': feature,
                'avg_score': round(score, 6)
            })
        
        # Simpan analisis TF-IDF
        self.analysis_results['tfidf_analysis'] = {
            'matrix_shape': self.tfidf_matrix.shape,
            'vocabulary_size': len(self.feature_names),
            'sparsity_percentage': round((1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100, 2),
            'top_features': top_features_data,
            'mean_tfidf_scores': {
                'mean': round(mean_scores.mean(), 6),
                'std': round(mean_scores.std(), 6),
                'min': round(mean_scores.min(), 6),
                'max': round(mean_scores.max(), 6)
            }
        }
        
        return True
    
    def analyze_similarity_calculation(self, sample_books: List[int] = None):
        """
        Analisis proses similarity calculation secara detail
        
        Args:
            sample_books (List[int]): Index buku yang akan dianalisis
        """
        print("\n" + "="*70)
        print("🔍 ANALISIS PROSES COSINE SIMILARITY CALCULATION")
        print("="*70)
        
        if sample_books is None:
            # Ambil 5 buku pertama sebagai sample
            sample_books = list(range(min(5, len(self.df))))
        
        print(f"📊 Menganalisis similarity untuk {len(sample_books)} buku sample...")
        
        # Hitung cosine similarity untuk semua buku
        print(f"\n🔄 Menghitung cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print(f"✅ Similarity Matrix berhasil dihitung:")
        print(f"   - Shape: {self.similarity_matrix.shape}")
        print(f"   - Min similarity: {self.similarity_matrix.min():.4f}")
        print(f"   - Max similarity: {self.similarity_matrix.max():.4f}")
        print(f"   - Mean similarity: {self.similarity_matrix.mean():.4f}")
        
        # Analisis statistik similarity (exclude diagonal)
        off_diagonal_mask = ~np.eye(self.similarity_matrix.shape[0], dtype=bool)
        off_diagonal_similarities = self.similarity_matrix[off_diagonal_mask]
        
        similarity_stats = {
            'mean_off_diagonal': round(off_diagonal_similarities.mean(), 6),
            'median_off_diagonal': round(np.median(off_diagonal_similarities), 6),
            'std_off_diagonal': round(off_diagonal_similarities.std(), 6),
            'min_off_diagonal': round(off_diagonal_similarities.min(), 6),
            'max_off_diagonal': round(off_diagonal_similarities.max(), 6),
            'percentiles': {
                'p25': round(np.percentile(off_diagonal_similarities, 25), 6),
                'p50': round(np.percentile(off_diagonal_similarities, 50), 6),
                'p75': round(np.percentile(off_diagonal_similarities, 75), 6),
                'p90': round(np.percentile(off_diagonal_similarities, 90), 6),
                'p95': round(np.percentile(off_diagonal_similarities, 95), 6),
                'p99': round(np.percentile(off_diagonal_similarities, 99), 6)
            }
        }
        
        print(f"\n📊 Statistik Similarity (off-diagonal):")
        print(f"   - Mean: {similarity_stats['mean_off_diagonal']:.6f}")
        print(f"   - Median: {similarity_stats['median_off_diagonal']:.6f}")
        print(f"   - Std: {similarity_stats['std_off_diagonal']:.6f}")
        print(f"   - P90: {similarity_stats['percentiles']['p90']:.6f}")
        print(f"   - P95: {similarity_stats['percentiles']['p95']:.6f}")
        print(f"   - Max: {similarity_stats['max_off_diagonal']:.6f}")
        
        # Simpan statistik similarity
        self.analysis_results['similarity_statistics'] = similarity_stats
        
        # Analisis detail untuk sample books
        print(f"\n📖 ANALISIS DETAIL UNTUK SAMPLE BOOKS:")
        print("-" * 70)
        
        for i, book_idx in enumerate(sample_books):
            book_title = self.df.iloc[book_idx]['judul_utama']
            print(f"\n{i+1}. Buku: '{book_title}' (Index: {book_idx})")
            
            # Ambil similarity scores untuk buku ini
            similarities = self.similarity_matrix[book_idx]
            
            # Urutkan berdasarkan similarity (exclude diri sendiri)
            similar_indices = np.argsort(similarities)[::-1][1:6]  # Top 5 excluding self
            
            print(f"   📊 Statistik similarity:")
            print(f"      - Mean: {similarities.mean():.4f}")
            print(f"      - Std:  {similarities.std():.4f}")
            print(f"      - Max:  {similarities.max():.4f} (dengan diri sendiri)")
            
            print(f"   🎯 Top 5 buku paling mirip:")
            for j, sim_idx in enumerate(similar_indices, 1):
                sim_title = self.df.iloc[sim_idx]['judul_utama']
                sim_score = similarities[sim_idx]
                print(f"      {j}. {sim_title[:40]:<40} (Score: {sim_score:.4f})")
        
        return True
    
    def detailed_similarity_breakdown(self, book1_idx: int, book2_idx: int, top_features: int = 10):
        """
        Breakdown detail similarity antara dua buku
        
        Args:
            book1_idx (int): Index buku pertama
            book2_idx (int): Index buku kedua
            top_features (int): Jumlah fitur teratas yang ditampilkan
        """
        print("\n" + "="*70)
        print("🔍 BREAKDOWN DETAIL SIMILARITY ANTARA DUA BUKU")
        print("="*70)
        
        book1_title = self.df.iloc[book1_idx]['judul_utama']
        book2_title = self.df.iloc[book2_idx]['judul_utama']
        
        print(f"📚 Buku 1: '{book1_title}' (Index: {book1_idx})")
        print(f"📚 Buku 2: '{book2_title}' (Index: {book2_idx})")
        
        # Ambil TF-IDF vectors untuk kedua buku
        vec1 = self.tfidf_matrix[book1_idx].toarray().flatten()
        vec2 = self.tfidf_matrix[book2_idx].toarray().flatten()
        
        # Hitung cosine similarity manual
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine_sim = dot_product / (norm1 * norm2)
        
        print(f"\n🧮 Perhitungan Cosine Similarity:")
        print(f"   - Dot product: {dot_product:.6f}")
        print(f"   - Norm buku 1: {norm1:.6f}")
        print(f"   - Norm buku 2: {norm2:.6f}")
        print(f"   - Cosine similarity: {cosine_sim:.6f}")
        
        # Verifikasi dengan sklearn
        sklearn_sim = self.similarity_matrix[book1_idx, book2_idx]
        print(f"   - Sklearn result: {sklearn_sim:.6f}")
        print(f"   - Difference: {abs(cosine_sim - sklearn_sim):.8f}")
        
        # Analisis fitur yang berkontribusi
        print(f"\n📊 Top {top_features} fitur yang berkontribusi pada similarity:")
        
        # Hitung kontribusi setiap fitur
        contributions = vec1 * vec2
        top_contrib_indices = contributions.argsort()[-top_features:][::-1]
        
        print(f"   {'Rank':<4} {'Feature':<25} {'TF-IDF 1':<10} {'TF-IDF 2':<10} {'Contribution':<12}")
        print(f"   {'-'*4} {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
        
        for i, idx in enumerate(top_contrib_indices, 1):
            feature = self.feature_names[idx]
            tfidf1 = vec1[idx]
            tfidf2 = vec2[idx]
            contrib = contributions[idx]
            
            if contrib > 0:  # Hanya tampilkan yang berkontribusi positif
                print(f"   {i:<4} {feature:<25} {tfidf1:<10.4f} {tfidf2:<10.4f} {contrib:<12.6f}")
        
        return cosine_sim
    
    def visualize_similarity_matrix(self, sample_size: int = 20, save_plot: bool = True):
        """
        Visualisasi similarity matrix
        
        Args:
            sample_size (int): Ukuran sample untuk visualisasi
            save_plot (bool): Apakah menyimpan plot
        """
        print(f"\n📊 Membuat visualisasi similarity matrix (sample {sample_size} buku)...")
        
        # Ambil sample dari similarity matrix
        sample_matrix = self.similarity_matrix[:sample_size, :sample_size]
        sample_titles = [self.df.iloc[i]['judul_utama'][:20] + "..." 
                        if len(self.df.iloc[i]['judul_utama']) > 20 
                        else self.df.iloc[i]['judul_utama'] 
                        for i in range(sample_size)]
        
        # Buat heatmap dengan colormap yang lebih kontras
        plt.figure(figsize=(14, 12))
        
        # Pilihan colormap yang lebih kontras
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'RdYlBu_r', 'coolwarm', 'seismic']
        selected_cmap = 'viridis'  # Colormap dengan kontras tinggi
        
        sns.heatmap(sample_matrix, 
                   xticklabels=sample_titles,
                   yticklabels=sample_titles,
                   annot=True,  # Tampilkan nilai similarity
                   fmt='.3f',   # Format 3 desimal
                   cmap=selected_cmap,
                   cbar_kws={'label': 'Cosine Similarity'},
                   square=True,  # Membuat cell berbentuk persegi
                   linewidths=0.5)  # Garis pemisah antar cell
        
        plt.title('Similarity Matrix Heatmap\n(Content-Based Filtering)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Books', fontsize=12, fontweight='bold')
        plt.ylabel('Books', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('similarity_matrix_heatmap.png', dpi=300, bbox_inches='tight')
            print("✅ Heatmap disimpan sebagai 'similarity_matrix_heatmap.png'")
        
        plt.show()
    
    def analyze_recommendation_process(self, target_book_title: str, top_n: int = 5):
        """
        Analisis proses rekomendasi secara step-by-step
        
        Args:
            target_book_title (str): Judul buku target
            top_n (int): Jumlah rekomendasi
        """
        print("\n" + "="*70)
        print("🎯 ANALISIS PROSES RECOMMENDATION ENGINE")
        print("="*70)
        
        # Cari buku berdasarkan judul
        target_book_title_lower = target_book_title.lower()
        matching_books = self.df[self.df['judul_utama'].str.lower().str.contains(target_book_title_lower, na=False)]
        
        if len(matching_books) == 0:
            print(f"❌ Buku '{target_book_title}' tidak ditemukan")
            return
        
        # Ambil buku pertama yang match
        target_idx = matching_books.index[0]
        actual_title = matching_books.iloc[0]['judul_utama']
        
        print(f"🎯 Target buku: '{actual_title}' (Index: {target_idx})")
        
        # Step 1: Ambil similarity scores
        print(f"\n📊 Step 1: Mengambil similarity scores...")
        similarities = self.similarity_matrix[target_idx]
        print(f"   - Total similarity scores: {len(similarities)}")
        print(f"   - Similarity dengan diri sendiri: {similarities[target_idx]:.4f}")
        
        # Step 2: Sorting dan filtering
        print(f"\n🔄 Step 2: Sorting berdasarkan similarity score...")
        sim_scores = list(enumerate(similarities))
        sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        print(f"   - Buku dengan similarity tertinggi:")
        for i, (idx, score) in enumerate(sim_scores_sorted[:3]):
            title = self.df.iloc[idx]['judul_utama']
            if idx == target_idx:
                print(f"     {i+1}. {title} (DIRI SENDIRI) - Score: {score:.4f}")
            else:
                print(f"     {i+1}. {title} - Score: {score:.4f}")
        
        # Step 3: Exclude diri sendiri dan ambil top_n
        print(f"\n✂️  Step 3: Exclude diri sendiri dan ambil top {top_n}...")
        recommendations = sim_scores_sorted[1:top_n+1]  # Skip yang pertama (diri sendiri)
        
        print(f"   📋 Hasil rekomendasi final:")
        for i, (idx, score) in enumerate(recommendations, 1):
            book_info = self.df.iloc[idx]
            print(f"\n   {i}. {book_info['judul_utama']}")
            print(f"      📊 Similarity Score: {score:.4f}")
            print(f"      👤 Pengarang: {book_info['tajuk_pengarang']}")
            print(f"      📂 Kategori: {book_info['kategori']}")
            print(f"      🌐 Bahasa: {book_info['bahasa']}")
            print(f"      📝 Deskripsi: {book_info['deskripsi'][:100]}...")
        
        # Simpan sample recommendations
        if not hasattr(self, 'analysis_results'):
            self.analysis_results = {}
        
        if 'sample_recommendations' not in self.analysis_results:
            self.analysis_results['sample_recommendations'] = []
        
        # Format recommendations untuk JSON
        formatted_recommendations = []
        for rec in recommendations:
            formatted_rec = {
                'rank': rec[0] + 1,  # rec[0] is index, convert to rank
                'similarity_score': round(rec[1], 6),
                'book_details': {
                    'judul': self.df.iloc[rec[0]]['judul_utama'],
                    'pengarang': self.df.iloc[rec[0]]['tajuk_pengarang'],
                    'kategori': self.df.iloc[rec[0]]['kategori'],
                    'bahasa': self.df.iloc[rec[0]]['bahasa'],
                    'deskripsi': self.df.iloc[rec[0]]['deskripsi'][:200] + "..." if len(self.df.iloc[rec[0]]['deskripsi']) > 200 else self.df.iloc[rec[0]]['deskripsi']
                }
            }
            formatted_recommendations.append(formatted_rec)
        
        self.analysis_results['sample_recommendations'].append({
            'target_book': target_book_title,
            'recommendations': formatted_recommendations
        })
        
        return recommendations
    
    def analyze_threshold_impact(self, thresholds: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25]):
        """
        Analisis dampak threshold terhadap jumlah rekomendasi
        
        Args:
            thresholds (List[float]): List threshold yang akan dianalisis
        """
        print("\n" + "="*70)
        print("🎯 ANALISIS DAMPAK THRESHOLD PADA SISTEM REKOMENDASI")
        print("="*70)
        
        threshold_analysis = []
        
        for threshold in thresholds:
            print(f"\n📊 Threshold: {threshold}")
            
            # Hitung berapa banyak pasangan buku yang memiliki similarity >= threshold
            above_threshold = (self.similarity_matrix >= threshold).sum() - len(self.df)  # Exclude diagonal
            total_pairs = len(self.df) * (len(self.df) - 1)  # Exclude diagonal
            percentage = (above_threshold / total_pairs) * 100
            
            # Hitung rata-rata jumlah rekomendasi per buku
            avg_recommendations = 0
            for i in range(len(self.df)):
                similarities = self.similarity_matrix[i]
                valid_recommendations = (similarities >= threshold).sum() - 1  # Exclude self
                avg_recommendations += valid_recommendations
            avg_recommendations /= len(self.df)
            
            print(f"   - Pasangan buku dengan similarity ≥ {threshold}: {above_threshold}")
            print(f"   - Persentase dari total pasangan: {percentage:.2f}%")
            print(f"   - Rata-rata rekomendasi per buku: {avg_recommendations:.1f}")
            
            threshold_analysis.append({
                'threshold': threshold,
                'pairs_above_threshold': int(above_threshold),
                'percentage': round(percentage, 2),
                'avg_recommendations_per_book': round(avg_recommendations, 1)
            })
        
        # Analisis coverage dan degree distribution
        coverage_analysis = []
        degree_distributions = []
        
        for threshold_data in threshold_analysis:
            threshold = threshold_data['threshold']
            
            # Coverage: berapa buku yang punya ≥1 tetangga di atas threshold
            books_with_neighbors = 0
            degree_list = []
            
            for i in range(len(self.df)):
                similarities = self.similarity_matrix[i]
                neighbors_count = (similarities >= threshold).sum() - 1  # Exclude self
                degree_list.append(neighbors_count)
                
                if neighbors_count > 0:
                    books_with_neighbors += 1
            
            coverage_percentage = (books_with_neighbors / len(self.df)) * 100
            
            coverage_analysis.append({
                'threshold': threshold,
                'books_with_neighbors': books_with_neighbors,
                'coverage_percentage': round(coverage_percentage, 2),
                'books_without_neighbors': len(self.df) - books_with_neighbors
            })
            
            degree_distributions.append({
                'threshold': threshold,
                'degree_stats': {
                    'mean': round(np.mean(degree_list), 2),
                    'median': round(np.median(degree_list), 2),
                    'std': round(np.std(degree_list), 2),
                    'min': int(np.min(degree_list)),
                    'max': int(np.max(degree_list)),
                    'zero_degree_count': int(np.sum(np.array(degree_list) == 0))
                },
                'degree_distribution': degree_list
            })
            
            print(f"   - Coverage: {books_with_neighbors}/{len(self.df)} buku ({coverage_percentage:.1f}%) punya tetangga")
            print(f"   - Degree stats: mean={np.mean(degree_list):.1f}, std={np.std(degree_list):.1f}, zero={np.sum(np.array(degree_list) == 0)}")
        
        # Simpan hasil analisis threshold
        self.analysis_results['threshold_analysis'] = threshold_analysis
        self.analysis_results['coverage_analysis'] = coverage_analysis
        self.analysis_results['degree_distributions'] = degree_distributions
        
        return threshold_analysis
    
    def export_analysis_results(self, export_formats: List[str] = ['json', 'csv', 'html']):
        """
        Export hasil analisis ke berbagai format
        
        Args:
            export_formats (List[str]): Format export yang diinginkan
        """
        print("\n" + "="*70)
        print("💾 EXPORT HASIL ANALISIS")
        print("="*70)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comprehensive analysis results
        export_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_books': len(self.df),
                'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
                'similarity_matrix_shape': self.similarity_matrix.shape if self.similarity_matrix is not None else None
            },
            'tfidf_analysis': self.analysis_results.get('tfidf_analysis', {}),
            'similarity_statistics': self.analysis_results.get('similarity_statistics', {}),
            'threshold_analysis': self.analysis_results.get('threshold_analysis', []),
            'coverage_analysis': self.analysis_results.get('coverage_analysis', []),
            'degree_distributions': self.analysis_results.get('degree_distributions', []),
            'sample_recommendations': self.analysis_results.get('sample_recommendations', [])
        }
        
        # Export ke JSON
        if 'json' in export_formats:
            json_filename = f"similarity_analysis_{timestamp}.json"
            # Convert numpy types to native Python types
            export_data_converted = convert_numpy_types(export_data)
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(export_data_converted, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON export: {json_filename}")
        
        # Export ke CSV (similarity matrix)
        if 'csv' in export_formats:
            csv_filename = f"similarity_matrix_{timestamp}.csv"
            similarity_df = pd.DataFrame(
                self.similarity_matrix,
                index=[f"Book_{i}" for i in range(len(self.df))],
                columns=[f"Book_{i}" for i in range(len(self.df))]
            )
            similarity_df.to_csv(csv_filename)
            print(f"✅ CSV export: {csv_filename}")
            
            # Export book details
            books_csv = f"book_details_{timestamp}.csv"
            book_details = self.df[['judul_utama', 'tajuk_pengarang', 'kategori', 'bahasa', 'subjek_topik']].copy()
            book_details.to_csv(books_csv, index=False)
            print(f"✅ Book details CSV: {books_csv}")
        
        # Export ke HTML report
        if 'html' in export_formats:
            html_filename = f"similarity_analysis_report_{timestamp}.html"
            self._generate_html_report(export_data, html_filename)
            print(f"✅ HTML report: {html_filename}")
        
        return export_data
    
    def _generate_html_report(self, data: Dict, filename: str):
        """Generate HTML report dari hasil analisis"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Similarity Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Similarity Analysis Report</h1>
                <p><strong>Generated:</strong> {data['metadata']['timestamp']}</p>
                <p><strong>Total Books:</strong> {data['metadata']['total_books']}</p>
                <p><strong>TF-IDF Features:</strong> {data['metadata']['tfidf_features']}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Threshold Analysis</h2>
                <table>
                    <tr>
                        <th>Threshold</th>
                        <th>Pairs Above Threshold</th>
                        <th>Percentage</th>
                        <th>Avg Recommendations per Book</th>
                        <th>Coverage (%)</th>
                        <th>Books Without Neighbors</th>
                    </tr>
        """
        
        for i, threshold_data in enumerate(data.get('threshold_analysis', [])):
            coverage_data = data.get('coverage_analysis', [])[i] if i < len(data.get('coverage_analysis', [])) else {}
            html_content += f"""
                    <tr>
                        <td>{threshold_data['threshold']}</td>
                        <td>{threshold_data['pairs_above_threshold']}</td>
                        <td>{threshold_data['percentage']}%</td>
                        <td>{threshold_data['avg_recommendations_per_book']}</td>
                        <td>{coverage_data.get('coverage_percentage', 'N/A')}%</td>
                        <td>{coverage_data.get('books_without_neighbors', 'N/A')}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Key Insights</h2>
                <div class="metric">
                    <strong>Recommendation Strategy:</strong> 
                    Lower thresholds (0.05-0.1) provide more recommendations but may include less relevant books.
                    Higher thresholds (0.2+) provide fewer but more relevant recommendations.
                </div>
                <div class="metric">
                    <strong>System Performance:</strong> 
                    The similarity matrix shows the relationship strength between all book pairs.
                    Most book pairs have low similarity, which is expected in a diverse dataset.
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def run_complete_analysis(self, processed_data_path: str):
        """
        Jalankan analisis lengkap similarity calculation
        
        Args:
            processed_data_path (str): Path ke data yang sudah dipreprocess
        """
        print("🚀 ANALISIS LENGKAP SIMILARITY CALCULATION")
        print("="*70)
        
        # Load data
        if not self.load_processed_data(processed_data_path):
            return False
        
        # Analisis TF-IDF
        self.analyze_tfidf_process(max_features=1000, show_top_features=15)
        
        # Analisis Similarity Calculation
        self.analyze_similarity_calculation(sample_books=[0, 1, 2, 3, 4])
        
        # Breakdown detail antara dua buku
        if len(self.df) >= 2:
            self.detailed_similarity_breakdown(0, 1, top_features=10)
        
        # Visualisasi
        self.visualize_similarity_matrix(sample_size=min(15, len(self.df)))
        
        # Analisis proses rekomendasi
        if len(self.df) > 0:
            first_book_title = self.df.iloc[0]['judul_utama']
            self.analyze_recommendation_process(first_book_title, top_n=5)
        
        # Analisis threshold impact
        self.analyze_threshold_impact()
        
        # Export hasil analisis
        self.export_analysis_results(['json', 'csv', 'html'])
        
        print("\n" + "="*70)
        print("✅ ANALISIS SIMILARITY CALCULATION SELESAI!")
        print("📁 File yang dihasilkan:")
        print("   - similarity_matrix_heatmap.png (Visualisasi)")
        print("   - similarity_analysis_[timestamp].json (Data lengkap)")
        print("   - similarity_matrix_[timestamp].csv (Matrix similarity)")
        print("   - book_details_[timestamp].csv (Detail buku)")
        print("   - similarity_analysis_report_[timestamp].html (Laporan HTML)")
        print("="*70)
        
        return True


def main():
    """Main function untuk menjalankan analisis"""
    analyzer = SimilarityAnalyzer()
    
    # Path ke data yang sudah dipreprocess
    processed_data_path = "data/processed_books.csv"
    
    # Jalankan analisis lengkap
    analyzer.run_complete_analysis(processed_data_path)


if __name__ == "__main__":
    main()
