"""Script Koneksi CCTV untuk Deteksi Kendaraan
Membantu menemukan dan menguji koneksi RTSP dari CCTV
"""

import os
import sys
import cv2
import yaml
import argparse
import time
import socket
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))


# Preset URL RTSP untuk berbagai merek CCTV
RTSP_PRESETS = {
    "hikvision": {
        "name": "Hikvision",
        "urls": [
            "rtsp://{user}:{passwd}@{ip}:554/Streaming/Channels/101",
            "rtsp://{user}:{passwd}@{ip}:554/Streaming/Channels/1",
            "rtsp://{user}:{passwd}@{ip}:554/ISAPI/streaming/channels/101",
        ],
        "default_user": "admin",
        "default_pass": "admin123",
        "default_port": 554,
    },
    "dahua": {
        "name": "Dahua",
        "urls": [
            "rtsp://{user}:{passwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
            "rtsp://{user}:{passwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
        ],
        "default_user": "admin",
        "default_pass": "admin",
        "default_port": 554,
    },
    "tplink": {
        "name": "TP-Link Tapo",
        "urls": [
            "rtsp://{user}:{passwd}@{ip}:554/stream1",
            "rtsp://{user}:{passwd}@{ip}:554/stream2",
        ],
        "default_user": "admin",
        "default_pass": "admin",
        "default_port": 554,
    },
    "generic": {
        "name": "Generic/ONVIF",
        "urls": [
            "rtsp://{user}:{passwd}@{ip}:554/live",
            "rtsp://{user}:{passwd}@{ip}:554/stream1",
            "rtsp://{user}:{passwd}@{ip}:554/0",
            "rtsp://{user}:{passwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
        ],
        "default_user": "admin",
        "default_pass": "admin",
        "default_port": 554,
    },
}


def scan_network(ip_prefix="192.168.1", port=554, timeout=0.5):
    """Scan jaringan untuk mencari device dengan port RTSP terbuka."""
    print(f"\n[SCAN] Scanning {ip_prefix}.0/24 port {port}...")
    print("[SCAN] Mencari device dengan port RTSP terbuka...\n")

    found = []

    def check_ip(ip):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            if result == 0:
                return ip
        except:
            pass
        return None

    ips = [f"{ip_prefix}.{i}" for i in range(1, 255)]

    with ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(check_ip, ips)
        for ip in results:
            if ip:
                found.append(ip)
                print(f"  [FOUND] {ip}:{port}")

    if not found:
        print("  [INFO] Tidak ditemukan device dengan port RTSP terbuka.")
        print("  [INFO] Coba port lain (8554, 80, 8080) atau cek IP CCTV.")

    return found


def test_rtsp(url, timeout=10, show=True, max_frames=100):
    """Test koneksi RTSP stream."""
    print(f"\n[TEST] Testing: {url}")
    print("[TEST] Menunggu koneksi...")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("[ERROR] Gagal membuka stream!")
        print("[INFO] Pastikan URL benar dan CCTV menyala.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"[OK] Stream terbuka!")
    print(f"     Resolusi: {width}x{height}")
    print(f"     FPS: {fps}")
    print(f"     Tekan 'q' untuk berhenti\n")

    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame tidak terbaca, stream terputus.")
            break

        frame_count += 1
        if frame_count > max_frames:
            break

        # Tampilkan info
        elapsed = time.time() - start
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        info = f"Frame: {frame_count} | FPS: {current_fps:.1f} | {width}x{height}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if show:
            cv2.imshow("CCTV Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    total_time = time.time() - start
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n[RESULTS] Total frame: {frame_count}")
    print(f"[RESULTS] Waktu: {total_time:.1f} detik")
    print(f"[RESULTS] FPS rata-rata: {avg_fps:.1f}")

    return True


def discover_cctv(ip, user="admin", password="admin123", port=554):
    """Coba berbagai URL RTSP pada IP tertentu."""
    print(f"\n[DISCOVER] Mencoba semua URL RTSP pada {ip}...")

    for brand_key, brand in RTSP_PRESETS.items():
        for url_template in brand["urls"]:
            url = url_template.format(user=user, passwd=password, ip=ip, port=port)
            print(f"  [{brand['name']}] {url}")

            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    print(f"  [FOUND] URL berfungsi! Resolusi: {width}x{height}")
                    return url
            cap.release()

    print("  [INFO] Tidak ditemukan URL yang berfungsi.")
    print("  [INFO] Coba username/password lain atau cek dokumentasi CCTV.")
    return None


def save_config(url, output_path="config/cctv.yaml"):
    """Simpan URL CCTV ke file konfigurasi."""
    config = {
        "cctv": {
            "rtsp_url": url,
            "source_type": "rtsp",
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n[OK] Konfigurasi tersimpan di: {output_path}")


def run_detection(url, output_path=None):
    """Jalankan deteksi kendaraan pada stream CCTV."""
    from detect_with_tracking import VehicleDetectionPipeline, load_config

    config = load_config()
    model_path = "D:/KULIAH/Tugas Akhir/tugasakhir/runs/detect/models/vehicle_detection/weights/best.pt"

    pipeline = VehicleDetectionPipeline(config, model_path=model_path)

    print(f"\n[RUN] Memulai deteksi pada: {url}")
    print("[RUN] Tekan 'q' untuk berhenti\n")

    pipeline.process_video(
        source=url,
        output_path=output_path,
        show=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Koneksi CCTV untuk Deteksi Kendaraan")
    parser.add_argument("--action", type=str, default="menu",
                       choices=["menu", "scan", "test", "discover", "run"],
                       help="Aksi yang dilakukan")
    parser.add_argument("--ip", type=str, default=None, help="IP Address CCTV")
    parser.add_argument("--url", type=str, default=None, help="URL RTSP langsung")
    parser.add_argument("--user", type=str, default="admin", help="Username CCTV")
    parser.add_argument("--password", type=str, default="admin123", help="Password CCTV")
    parser.add_argument("--port", type=int, default=554, help="Port RTSP")
    parser.add_argument("--prefix", type=str, default="192.168.1", help="IP prefix untuk scan")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--no-show", action="store_true", help="Tidak tampilkan preview")
    args = parser.parse_args()

    if args.action == "scan":
        scan_network(ip_prefix=args.prefix, port=args.port)

    elif args.action == "test":
        if not args.url:
            print("[ERROR] URL harus diisi! Contoh: --url rtsp://admin:pass@192.168.1.100:554/stream1")
            return
        test_rtsp(args.url, show=not args.no_show)

    elif args.action == "discover":
        if not args.ip:
            print("[ERROR] IP harus diisi! Contoh: --ip 192.168.1.100")
            return
        url = discover_cctv(args.ip, user=args.user, password=args.password, port=args.port)
        if url:
            save_config(url)

    elif args.action == "run":
        url = args.url
        if not url:
            # Coba load dari config
            config_path = "config/cctv.yaml"
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                url = cfg.get("cctv", {}).get("rtsp_url")
                if url:
                    print(f"[INFO] URL dari config: {url}")

        if not url:
            print("[ERROR] URL harus diisi atau simpan config dulu dengan --action discover")
            return

        run_detection(url, output_path=args.output)

    else:
        # Menu interaktif
        print("\n" + "=" * 60)
        print("CCTV CONNECTION TOOL - Deteksi Kendaraan")
        print("=" * 60)
        print("""
Pilihan:
  1. Scan jaringan (cari CCTV)
  2. Test URL RTSP
  3. Auto-discover URL CCTV
  4. Jalankan deteksi dari CCTV
  5. Jalankan deteksi dari file video
  6. Keluar
        """)

        choice = input("Pilihan [1-6]: ").strip()

        if choice == "1":
            prefix = input("IP prefix (default: 192.168.1): ").strip() or "192.168.1"
            scan_network(ip_prefix=prefix)

        elif choice == "2":
            url = input("URL RTSP: ").strip()
            if url:
                test_rtsp(url)

        elif choice == "3":
            ip = input("IP Address CCTV: ").strip()
            user = input("Username (default: admin): ").strip() or "admin"
            password = input("Password (default: admin123): ").strip() or "admin123"
            if ip:
                url = discover_cctv(ip, user=user, password=password)
                if url:
                    save = input("\nSimpan konfigurasi? [y/n]: ").strip().lower()
                    if save == "y":
                        save_config(url)

        elif choice == "4":
            url = input("URL RTSP: ").strip()
            if url:
                run_detection(url)

        elif choice == "5":
            path = input("Path video: ").strip()
            if path:
                run_detection(path)

        elif choice == "6":
            print("Selesai.")


if __name__ == "__main__":
    main()
