"""
Script untuk menjalankan analisis similarity calculation
"""

from similarity_analysis import SimilarityAnalyzer

def main():
    print("🔍 MENJALANKAN ANALISIS SIMILARITY CALCULATION")
    print("="*60)
    
    # Inisialisasi analyzer
    analyzer = SimilarityAnalyzer()
    
    # Path ke data yang sudah dipreprocess
    processed_data_path = "data/processed_books.csv"
    
    # Jalankan analisis lengkap
    success = analyzer.run_complete_analysis(processed_data_path)
    
    if success:
        print("\n🎯 ANALISIS SELESAI!")
        print("Anda dapat melihat:")
        print("1. Proses TF-IDF vectorization")
        print("2. Similarity matrix calculation")
        print("3. Breakdown detail similarity antara buku")
        print("4. Visualisasi heatmap (similarity_matrix_heatmap.png)")
        print("5. Step-by-step recommendation process")
    else:
        print("❌ Analisis gagal!")

if __name__ == "__main__":
    main()
