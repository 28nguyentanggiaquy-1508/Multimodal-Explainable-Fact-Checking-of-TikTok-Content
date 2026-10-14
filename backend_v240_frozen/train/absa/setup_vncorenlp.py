from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from config_ate_v2 import VNCORENLP_DIR


FILES = {
    "VnCoreNLP-1.2.jar": (
        "https://raw.githubusercontent.com/"
        "vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar"
    ),
    "models/wordsegmenter/vi-vocab": (
        "https://raw.githubusercontent.com/"
        "vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab"
    ),
    "models/wordsegmenter/wordsegmenter.rdr": (
        "https://raw.githubusercontent.com/"
        "vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
    ),
}


def check_java():
    java = shutil.which("java")

    if not java:
        raise RuntimeError(
            "Không tìm thấy Java trong PATH.\n"
            "VnCoreNLP yêu cầu Java 8+.\n"
            "Hãy cài Java/JDK rồi mở terminal mới và kiểm tra: java -version"
        )

    result = subprocess.run(
        [java, "-version"],
        capture_output=True,
        text=True,
    )

    version_text = (
        result.stderr
        or result.stdout
        or ""
    ).strip()

    print("[OK] Java:")
    print(version_text.splitlines()[0] if version_text else java)


def download_file(url: str, destination: Path):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"[SKIP] Đã có: {destination} "
            f"({destination.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return

    temp_file = destination.with_suffix(
        destination.suffix + ".part"
    )

    if temp_file.exists():
        temp_file.unlink()

    print(f"[DOWNLOAD] {url}")
    print(f"       -> {destination}")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response, open(
            temp_file,
            "wb",
        ) as output:
            total = response.headers.get(
                "Content-Length"
            )

            total = (
                int(total)
                if total
                else None
            )

            downloaded = 0
            chunk_size = 1024 * 1024

            while True:
                chunk = response.read(
                    chunk_size
                )

                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = (
                        downloaded
                        / total
                        * 100
                    )

                    print(
                        f"\r       "
                        f"{downloaded / 1024 / 1024:.1f} MB "
                        f"/ {total / 1024 / 1024:.1f} MB "
                        f"({percent:.1f}%)",
                        end="",
                    )
                else:
                    print(
                        f"\r       "
                        f"{downloaded / 1024 / 1024:.1f} MB",
                        end="",
                    )

            print()

        temp_file.replace(
            destination
        )

    except Exception:
        if temp_file.exists():
            temp_file.unlink()

        raise


def verify_files(target: Path):
    missing = []

    for relative_path in FILES:
        file_path = (
            target
            / relative_path
        )

        if (
            not file_path.exists()
            or file_path.stat().st_size == 0
        ):
            missing.append(
                str(file_path)
            )

    if missing:
        raise RuntimeError(
            "VnCoreNLP tải chưa đủ file:\n"
            + "\n".join(missing)
        )

    print("[OK] Đủ file VnCoreNLP cho word segmentation.")


def test_vncorenlp(target: Path):
    try:
        import py_vncorenlp
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu py_vncorenlp.\n"
            "Chạy: pip install py_vncorenlp"
        ) from exc

    print("[TEST] Loading VnCoreNLP wseg...")

    model = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=str(
            target.resolve()
        ),
    )

    sample = (
        "Màn hình điện thoại đẹp nhưng pin hơi yếu."
    )

    output = model.word_segment(
        sample
    )

    print("[OK] VnCoreNLP hoạt động.")
    print("Input :", sample)
    print("Output:", output)


def main():
    target = Path(
        VNCORENLP_DIR
    ).resolve()

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "VnCoreNLP directory:",
        target,
    )

    check_java()

    for relative_path, url in FILES.items():
        download_file(
            url,
            target / relative_path,
        )

    verify_files(
        target
    )

    test_vncorenlp(
        target
    )

    print()
    print("=" * 60)
    print("SETUP VNCORENLP THÀNH CÔNG")
    print("=" * 60)
    print(
        "Bước tiếp theo:"
    )
    print(
        "python backend\\train\\absa\\prepare_ate_v2_data.py"
    )


if __name__ == "__main__":
    main()
