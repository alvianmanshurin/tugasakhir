"""
Setup LabelImg untuk anotasi gambar
"""
import os
import subprocess
import sys

def install_labelimg():
    """Instal LabelImg untuk anotasi"""
    print("=== INSTALASI LABELIMG ===")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "labelimg"])
        print("LabelImg berhasil diinstal")
        print("Gunakan: labelImg untuk menjalankan")
    except Exception as e:
        print(f"Gagal menginstal LabelImg: {e}")

def main():
    install_labelimg()

if __name__ == "__main__":
    main()
