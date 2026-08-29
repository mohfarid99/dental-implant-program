# archive

Kode lama yang tidak lagi dipakai aplikasi. Dipindah ke sini agar root bersih —
semuanya masih ada di git history, jadi aman dibuka atau dikembalikan kapan saja.

## legacy-versions/
`main.py` … `main45.py` (termasuk `main36-reserved.py`) — 47 iterasi aplikasi
desktop sebelum versi yang dipakai sekarang. Versi live ada di root:
`main46.py` (GPU) dan `main46CPU.py` (CPU-only), sesuai target kedua file `.spec`.

## prototypes/
Eksperimen awal dan file yang tidak direferensikan aplikasi live:

| file | keterangan |
|---|---|
| `main.ui`, `main2.ui` | sumber Qt Designer untuk `main.py` / `main2.py` yang sudah usang |
| `mainwindow.py` | hasil pyuic5 yang isinya cuma header, tanpa kelas |
| `test.py` | scratch vispy/PyQt |
| `FastAPI.py` | percobaan FastAPI awal, digantikan `web/backend/app.py` |
| `removeDCM.py` | utilitas sekali pakai |
| `skeleton.py`, `.ipynb`, `.png` | eksperimen skeletonisasi |
| `convert ui to py.txt` | catatan perintah `pyuic5` untuk `main2.ui` |

## implant_database - original.csv
Cadangan database implan sebelum diedit. Yang dipakai aplikasi adalah
`implant_database.csv` di root.

---

**Jangan pindahkan file dari root ke sini:** `main46.py` mengimpor
`recommendation.py` (`from recommendation import Ui_MainWindow`), dan baik
aplikasi desktop maupun `web/backend/app.py` mencari `classification.pt`,
`segmentation.pt`, dan `implant_database.csv` tepat di root.
