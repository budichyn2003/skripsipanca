"""
Script untuk menjalankan preprocessing sistem rekomendasi buku
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))

from preprocessing.book_preprocessing import BookPreprocessor

def main():
    print("🚀 Memulai Preprocessing Sistem Rekomendasi Buku")
    print("=" * 50)
    
    # Path file
    input_file = "eaa.csv"
    output_file = "data/processed_books.csv"
    
    # Pastikan folder data ada
    os.makedirs("data", exist_ok=True)
    
    # Inisialisasi preprocessor
    preprocessor = BookPreprocessor()
    
    # Jalankan preprocessing lengkap
    df_processed = preprocessor.run_full_preprocessing(input_file, max_features=5000)
    
    if df_processed is not None:
        # Simpan hasil
        preprocessor.save_processed_data(df_processed, output_file)
        
        # Tampilkan statistik
        print("\n📊 STATISTIK HASIL PREPROCESSING:")
        print(f"Total buku: {len(df_processed)}")
        print(f"Kolom yang tersedia: {list(df_processed.columns)}")
        
        # Tampilkan contoh
        print("\n📖 CONTOH HASIL:")
        sample = df_processed.iloc[0]
        print(f"Judul: {sample['judul_utama']}")
        print(f"Original: {sample['combined_features'][:100]}...")
        print(f"Processed: {sample['processed_features'][:100]}...")
        
        print(f"\n✅ Preprocessing selesai! File disimpan di: {output_file}")
        print("🎯 Dataset siap untuk sistem rekomendasi content-based filtering")
        
    else:
        print("❌ Preprocessing gagal!")

if __name__ == "__main__":
    main()
