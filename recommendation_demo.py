"""
Demo Interface untuk Sistem Rekomendasi Buku Content-Based Filtering
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.content_based_recommender import ContentBasedRecommender

def print_separator(title=""):
    """Print separator dengan title"""
    print("\n" + "="*60)
    if title:
        print(f" {title} ")
        print("="*60)

def print_book_info(book, show_score=False):
    """Print informasi buku dengan format yang rapi"""
    print(f"📚 {book['judul']}")
    print(f"   👤 Pengarang: {book['pengarang']}")
    print(f"   🏢 Penerbit: {book['penerbit']} ({book['tahun']})")
    print(f"   📂 Kategori: {book['kategori']}")
    print(f"   🌐 Bahasa: {book['bahasa']}")
    print(f"   🏷️  Subjek: {book['subjek_topik']}")
    if show_score and 'similarity_score' in book:
        print(f"   ⭐ Similarity Score: {book['similarity_score']}")
    print(f"   📝 Deskripsi: {book['deskripsi'][:150]}...")
    print()

def demo_recommendation_by_title(recommender):
    """Demo rekomendasi berdasarkan judul buku"""
    print_separator("DEMO 1: REKOMENDASI BERDASARKAN JUDUL BUKU")
    
    # Contoh judul buku yang ada di dataset
    test_titles = [
        "Accounting & auditing research",
        "Riset akuntansi", 
        "Advanced accounting"
    ]
    
    for title in test_titles:
        print(f"\n🔍 Mencari rekomendasi untuk buku: '{title}'")
        print("-" * 50)
        
        recommendations = recommender.get_book_recommendations(title, top_n=3)
        
        if recommendations:
            print(f"✅ Ditemukan {len(recommendations)} rekomendasi:")
            for rec in recommendations:
                print(f"\n{rec['rank']}. ", end="")
                print_book_info(rec, show_score=True)
        else:
            print("❌ Tidak ditemukan rekomendasi")

def demo_search_by_keyword(recommender):
    """Demo pencarian berdasarkan keyword"""
    print_separator("DEMO 2: PENCARIAN BERDASARKAN KEYWORD")
    
    keywords = ["akuntansi", "accounting", "financial", "audit"]
    
    for keyword in keywords:
        print(f"\n🔍 Mencari buku dengan keyword: '{keyword}'")
        print("-" * 50)
        
        search_results = recommender.search_books_by_keyword(keyword, top_n=3)
        
        if search_results:
            print(f"✅ Ditemukan {len(search_results)} buku:")
            for i, book in enumerate(search_results, 1):
                print(f"\n{i}. ", end="")
                print_book_info(book)
        else:
            print("❌ Tidak ditemukan buku")

def demo_recommendation_by_features(recommender):
    """Demo rekomendasi berdasarkan fitur"""
    print_separator("DEMO 3: REKOMENDASI BERDASARKAN FITUR")
    
    feature_sets = [
        {"kategori": "Accounting", "bahasa": "inggris"},
        {"bahasa": "indonesia"},
        {"subjek_topik": "akuntansi"}
    ]
    
    for i, features in enumerate(feature_sets, 1):
        print(f"\n🔍 Demo {i} - Fitur: {features}")
        print("-" * 50)
        
        recommendations = recommender.get_recommendations_by_features(features, top_n=3)
        
        if recommendations:
            print(f"✅ Ditemukan {len(recommendations)} rekomendasi:")
            for rec in recommendations:
                print(f"\n{rec['rank']}. ", end="")
                print_book_info(rec)
        else:
            print("❌ Tidak ditemukan rekomendasi")

def interactive_mode(recommender):
    """Mode interaktif untuk user"""
    print_separator("MODE INTERAKTIF")
    print("Pilih opsi:")
    print("1. Rekomendasi berdasarkan judul buku")
    print("2. Pencarian berdasarkan keyword")
    print("3. Rekomendasi berdasarkan kategori/bahasa")
    print("4. Keluar")
    
    while True:
        try:
            choice = input("\nPilih opsi (1-4): ").strip()
            
            if choice == "1":
                title = input("Masukkan judul buku: ").strip()
                if title:
                    print(f"\n🔍 Mencari rekomendasi untuk: '{title}'")
                    recommendations = recommender.get_book_recommendations(title, top_n=5)
                    
                    if recommendations:
                        print(f"✅ Ditemukan {len(recommendations)} rekomendasi:")
                        for rec in recommendations:
                            print(f"\n{rec['rank']}. ", end="")
                            print_book_info(rec, show_score=True)
                    else:
                        print("❌ Tidak ditemukan rekomendasi")
            
            elif choice == "2":
                keyword = input("Masukkan keyword pencarian: ").strip()
                if keyword:
                    print(f"\n🔍 Mencari buku dengan keyword: '{keyword}'")
                    search_results = recommender.search_books_by_keyword(keyword, top_n=5)
                    
                    if search_results:
                        print(f"✅ Ditemukan {len(search_results)} buku:")
                        for i, book in enumerate(search_results, 1):
                            print(f"\n{i}. ", end="")
                            print_book_info(book)
                    else:
                        print("❌ Tidak ditemukan buku")
            
            elif choice == "3":
                print("\nMasukkan kriteria (kosongkan jika tidak ingin filter):")
                kategori = input("Kategori (contoh: Accounting): ").strip()
                bahasa = input("Bahasa (indonesia/inggris): ").strip()
                subjek = input("Subjek topik: ").strip()
                
                features = {}
                if kategori: features['kategori'] = kategori
                if bahasa: features['bahasa'] = bahasa
                if subjek: features['subjek_topik'] = subjek
                
                if features:
                    print(f"\n🔍 Mencari buku dengan kriteria: {features}")
                    recommendations = recommender.get_recommendations_by_features(features, top_n=5)
                    
                    if recommendations:
                        print(f"✅ Ditemukan {len(recommendations)} rekomendasi:")
                        for rec in recommendations:
                            print(f"\n{rec['rank']}. ", end="")
                            print_book_info(rec)
                    else:
                        print("❌ Tidak ditemukan rekomendasi")
                else:
                    print("❌ Tidak ada kriteria yang dimasukkan")
            
            elif choice == "4":
                print("👋 Terima kasih telah menggunakan sistem rekomendasi!")
                break
            
            else:
                print("❌ Pilihan tidak valid. Silakan pilih 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Program dihentikan oleh user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function untuk menjalankan demo"""
    print("🚀 SISTEM REKOMENDASI BUKU CONTENT-BASED FILTERING")
    print("=" * 60)
    print("📚 Memuat sistem rekomendasi...")
    
    # Inisialisasi recommender
    recommender = ContentBasedRecommender()
    
    # Path ke data yang sudah dipreprocess
    processed_data_path = "data/processed_books.csv"
    
    # Build sistem rekomendasi
    if not recommender.build_recommendation_system(processed_data_path):
        print("❌ Gagal membangun sistem rekomendasi!")
        return
    
    # Simpan model untuk penggunaan selanjutnya
    model_path = "model/book_recommender_model.pkl"
    recommender.save_model(model_path)
    
    print("\n🎯 Sistem rekomendasi siap digunakan!")
    
    # Jalankan demo otomatis
    demo_recommendation_by_title(recommender)
    demo_search_by_keyword(recommender)
    demo_recommendation_by_features(recommender)
    
    # Mode interaktif
    try:
        interactive_mode(recommender)
    except KeyboardInterrupt:
        print("\n👋 Program selesai!")

if __name__ == "__main__":
    main()
