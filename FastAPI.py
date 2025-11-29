from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

app = FastAPI()


# Halaman utama: tampilkan form HTML
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>Contoh FastAPI Web</title>
        </head>
        <body>
            <h1>Contoh Web dengan FastAPI + Python</h1>
            <p>Masukkan angka, nanti akan dikali 2 di server (Python).</p>

            <form action="/hitung" method="post">
                <label>Angka:</label>
                <input type="number" name="angka" step="any" required>
                <button type="submit">Hitung</button>
            </form>
        </body>
    </html>
    """
    return html_content


# Route untuk proses form (eksekusi Python)
@app.post("/hitung", response_class=HTMLResponse)
def hitung(angka: float = Form(...)):
    hasil = angka * 2  # proses pakai Python

    html_content = f"""
    <html>
        <head>
            <title>Hasil Perhitungan</title>
        </head>
        <body>
            <h1>Hasil Perhitungan</h1>
            <p>Input: {angka}</p>
            <p>Output (dikali 2): {hasil}</p>

            <a href="/">Kembali ke form</a>
        </body>
    </html>
    """
    return html_content
