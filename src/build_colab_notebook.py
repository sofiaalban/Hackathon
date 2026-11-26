"""Genera notebooks/00_colab_verificar_y_app.ipynb."""
import re
import nbformat as nbf

src = open("src/build_notebook.py").read()
SETUP = re.search(r'SETUP = r"""(.*?)"""', src, re.S).group(1).strip()

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# ✅ Verificación de entregables + App Streamlit en Google Colab
Ejecuta las celdas **en orden** (`Entorno de ejecución → Ejecutar todas` también sirve).

1. **Configuración**: sube `hack_entrega.zip` cuando lo pida (o pon la URL de tu repo en `REPO_URL`).
2. **Verificación**: estructura, formato de `submission.csv`, reproducibilidad del modelo y F1-Macro en CV.
3. *(Opcional)* re-ejecutar el notebook principal completo (~5–8 min).
4. **App Streamlit** con enlace público para abrirla en el navegador.
""")
code(SETUP)
code(r"""
# Dependencias (Colab ya trae pandas, numpy, scikit-learn, matplotlib, seaborn)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit>=1.50", "plotly"], check=True)
print("listo")
""")
md("## 2. Verificación automática")
code(r"""
r = subprocess.run([sys.executable, "verificar.py"], capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-2000:])
""")
md(r"""
## 3. (Opcional) Re-ejecutar el notebook principal completo
Genera `notebooks/01_eda_y_modelado_ejecutado.ipynb` y vuelve a escribir `entregas/submission.csv`.
Para verlo celda por celda en Colab: `Archivo → Subir notebook` y elige `notebooks/01_eda_y_modelado.ipynb` (su primera celda hace la misma configuración).
""")
code(r"""
EJECUTAR_NOTEBOOK = False   # cambia a True para correrlo (~5-8 min)
if EJECUTAR_NOTEBOOK:
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=1800",
                        "--output", "01_eda_y_modelado_ejecutado.ipynb",
                        "notebooks/01_eda_y_modelado.ipynb"], capture_output=True, text=True)
    print(r.stderr[-1500:])
    subprocess.run([sys.executable, "verificar.py"])
""")
md(r"""
## 4. Levantar la app de Streamlit
Se abre un túnel gratuito de Cloudflare (sin registro). Haz clic en el enlace `https://….trycloudflare.com` que aparece.
La app queda activa mientras esta sesión de Colab siga abierta. Para detenerla, ejecuta la última celda.
""")
code(r"""
import time, re, os, stat, urllib.request

# 1) Streamlit en segundo plano
st_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py",
                            "--server.port", "8501", "--server.headless", "true"],
                           stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=2); break
    except Exception:
        time.sleep(1)
print("Streamlit corriendo en el puerto 8501")

# 2) Túnel público con cloudflared
if not os.path.exists("cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "cloudflared")
    os.chmod("cloudflared", os.stat("cloudflared").st_mode | stat.S_IEXEC)
cf_proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
                           stdout=open("cloudflared.log", "w"), stderr=subprocess.STDOUT)
url = None
for _ in range(60):
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("cloudflared.log").read())
    if m:
        url = m.group(0); break
    time.sleep(1)
print("👉 Abre la app aquí:", url or "no se obtuvo URL; revisa cloudflared.log o usa la celda siguiente")
""")
code(r"""
# Alternativa sin túnel: abre la app con el proxy de Colab (funciona en el mismo navegador)
if IN_COLAB:
    from google.colab.output import serve_kernel_port_as_window
    serve_kernel_port_as_window(8501)
""")
code(r"""
# Si algo falla, revisa los logs:
print(open("streamlit.log").read()[-2000:])
""")
code(r"""
# Detener la app y el túnel
for p in ["st_proc", "cf_proc"]:
    if p in globals():
        globals()[p].terminate()
print("detenido")
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, "notebooks/00_colab_verificar_y_app.ipynb")
print("ok")
