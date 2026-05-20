from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from src.skin_patterns.config import PipelineConfig
from src.skin_patterns.pipeline import run_pipeline


st.set_page_config(page_title="Patrones de piel", layout="wide")

st.title("Patrones de piel en dermatologia")
st.caption("Analisis no supervisado con vision por computadora para agrupar lesiones por similitud visual.")

with st.sidebar:
    st.header("Configuracion")
    method = st.selectbox("Metodo", ["kmeans", "fuzzy", "gmm", "dbscan"])
    clusters = st.slider("Clusters", min_value=2, max_value=8, value=4)
    image_size = st.select_slider("Tamano de imagen", options=[128, 160, 224, 256], value=224)
    pca_components = st.slider("Componentes PCA", min_value=2, max_value=24, value=12)

uploaded_files = st.file_uploader(
    "Carga imagenes dermatologicas",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Carga al menos dos imagenes para ejecutar el agrupamiento.")
    st.stop()

if len(uploaded_files) < 2:
    st.warning("Se necesitan al menos dos imagenes para evaluar clusters.")
    st.stop()

with TemporaryDirectory() as tmpdir:
    input_dir = Path(tmpdir)
    for uploaded_file in uploaded_files:
        target = input_dir / uploaded_file.name
        target.write_bytes(uploaded_file.getbuffer())

    config = PipelineConfig(
        image_size=(image_size, image_size),
        clusters=min(clusters, len(uploaded_files)),
        pca_components=min(pca_components, len(uploaded_files)),
    )

    with st.spinner("Procesando imagenes y calculando clusters..."):
        table, result = run_pipeline(input_dir, method=method, config=config, save_artifacts=False)

    metric_cols = st.columns(3)
    metric_names = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    for col, name in zip(metric_cols, metric_names):
        value = result.metrics[name]
        col.metric(name.replace("_", " ").title(), "N/A" if value is None else f"{value:.3f}")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Mapa PCA")
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(table["pc1"], table["pc2"], c=table["cluster"], cmap="tab10", s=80)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        st.pyplot(fig)

    with right:
        st.subheader("Resultados")
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Imagenes agrupadas")
    for cluster_id in sorted(pd.unique(table["cluster"])):
        st.markdown(f"**Cluster {cluster_id}**")
        cluster_files = table.loc[table["cluster"] == cluster_id, "image"].tolist()
        columns = st.columns(min(4, len(cluster_files)))
        for col, image_name in zip(columns, cluster_files):
            col.image(Image.open(input_dir / image_name), caption=image_name, use_container_width=True)
