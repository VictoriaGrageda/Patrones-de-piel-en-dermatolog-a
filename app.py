import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from PIL import ImageOps

from src.skin_patterns.config import MODELS_DIR, PipelineConfig, RAW_DATA_DIR, REPORTS_DIR
from src.skin_patterns.ham10000 import SmallDermCnn
from src.skin_patterns.pipeline import run_pipeline


HAM10000_LABELS = {
    "akiec": "Queratosis actinica / enfermedad de Bowen",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesiones benignas tipo queratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanociticos",
    "vasc": "Lesiones vasculares",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_ham10000_model(model_path: str) -> tuple[SmallDermCnn, dict[str, int], dict[int, str], int]:
    checkpoint = torch.load(model_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {int(key): value for key, value in checkpoint["index_to_label"].items()}
    image_size = int(checkpoint.get("config", {}).get("image_size", 128))

    model = SmallDermCnn(num_classes=len(label_to_index))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_to_index, index_to_label, image_size


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def predict_ham10000(image: Image.Image) -> pd.DataFrame:
    model_path = MODELS_DIR / "ham10000_cnn_from_scratch.pt"
    model, _, index_to_label, image_size = load_ham10000_model(str(model_path))
    tensor = image_to_tensor(image, image_size)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)

    rows = []
    for index, probability in enumerate(probabilities.tolist()):
        label = index_to_label[index]
        rows.append(
            {
                "clase": label,
                "descripcion": HAM10000_LABELS.get(label, label),
                "probabilidad": probability,
            }
        )
    return pd.DataFrame(rows).sort_values("probabilidad", ascending=False).reset_index(drop=True)


def find_ham10000_metadata() -> Path:
    default_path = RAW_DATA_DIR / "HAM10000" / "HAM10000_metadata.csv"
    if default_path.exists():
        return default_path

    matches = sorted(RAW_DATA_DIR.rglob("HAM10000_metadata.csv"))
    return matches[0] if matches else default_path


def render_dataset_summary(metadata_path: Path) -> None:
    st.subheader("Dataset de entrenamiento")

    if not metadata_path.exists():
        st.info("Dataset pendiente: coloca HAM10000 dentro de `data/raw/HAM10000/` para mostrar sus datos.")
        st.code(
            "data/raw/HAM10000/\n"
            "|-- HAM10000_metadata.csv\n"
            "|-- HAM10000_images_part_1/\n"
            "`-- HAM10000_images_part_2/",
            language="text",
        )
        st.caption("Cuando ese archivo exista, esta pantalla mostrara cantidad de imagenes y distribucion por clase.")
        return

    metadata = pd.read_csv(metadata_path)
    image_count = len(metadata)
    class_counts = metadata["dx"].value_counts().rename_axis("clase").reset_index(name="imagenes")
    class_counts["descripcion"] = class_counts["clase"].map(HAM10000_LABELS)

    cols = st.columns(4)
    cols[0].metric("Imagenes en metadata", f"{image_count:,}")
    cols[1].metric("Clases", metadata["dx"].nunique())
    cols[2].metric("Pacientes", metadata["lesion_id"].nunique() if "lesion_id" in metadata else "N/A")
    cols[3].metric("Archivo", metadata_path.name)

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(class_counts, use_container_width=True, hide_index=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(class_counts["clase"], class_counts["imagenes"], color="#3b82f6")
        ax.set_xlabel("Clase")
        ax.set_ylabel("Imagenes")
        ax.set_title("Distribucion de clases HAM10000")
        st.pyplot(fig)


def render_training_results() -> None:
    st.subheader("Resultados del entrenamiento")

    history_path = REPORTS_DIR / "ham10000_training_history.csv"
    report_path = REPORTS_DIR / "ham10000_test_report.json"
    matrix_path = REPORTS_DIR / "ham10000_confusion_matrix.csv"

    if not history_path.exists():
        st.info("Entrenamiento pendiente: todavia no existe `reports/ham10000_training_history.csv`.")
        st.code(
            "python train_ham10000.py --data-dir data/raw/HAM10000 "
            "--metadata data/raw/HAM10000/HAM10000_metadata.csv --epochs 20 --batch-size 32",
            language="bash",
        )
        st.caption("Despues de entrenar, aqui apareceran las curvas de loss, macro F1, reporte por clase y matriz de confusion.")
        return

    history = pd.read_csv(history_path)
    last = history.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Epocas", int(last["epoch"]))
    cols[1].metric("Train accuracy", f"{last['train_accuracy']:.3f}")
    cols[2].metric("Val accuracy", f"{last['val_accuracy']:.3f}")
    cols[3].metric("Val macro F1", f"{last['val_macro_f1']:.3f}")

    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["epoch"], history["train_loss"], label="train loss")
        ax.plot(history["epoch"], history["val_loss"], label="val loss")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["epoch"], history["train_macro_f1"], label="train macro F1")
        ax.plot(history["epoch"], history["val_macro_f1"], label="val macro F1")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Macro F1")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    st.dataframe(history, use_container_width=True, hide_index=True)

    if report_path.exists():
        report = read_json(report_path)
        test_rows = []
        for label in HAM10000_LABELS:
            if label in report:
                test_rows.append(
                    {
                        "clase": label,
                        "precision": report[label]["precision"],
                        "recall": report[label]["recall"],
                        "f1": report[label]["f1-score"],
                        "soporte": report[label]["support"],
                    }
                )
        if test_rows:
            st.subheader("Reporte por clase en prueba")
            st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

    if matrix_path.exists():
        st.subheader("Matriz de confusion")
        matrix = pd.read_csv(matrix_path, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Clase real")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, int(matrix.iat[row, col]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
        st.pyplot(fig)


def render_prediction_view() -> None:
    st.subheader("Reconocimiento de una imagen")
    model_path = MODELS_DIR / "ham10000_cnn_from_scratch.pt"

    if not model_path.exists():
        st.info("Primero entrena el modelo HAM10000 para habilitar el reconocimiento de una imagen.")
        st.code(
            "python train_ham10000.py --epochs 1 --batch-size 32 --image-size 64 --max-samples 1000",
            language="bash",
        )
        return

    uploaded_file = st.file_uploader(
        "Carga una imagen dermatologica",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="ham10000_prediction_upload",
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    predictions = predict_ham10000(image)
    top = predictions.iloc[0]

    left, right = st.columns([0.8, 1.2])
    with left:
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    with right:
        st.metric("Prediccion principal", f"{top['clase']} ({top['probabilidad']:.1%})")
        st.write(top["descripcion"])
        st.dataframe(
            predictions.assign(probabilidad=predictions["probabilidad"].map(lambda value: f"{value:.2%}")),
            use_container_width=True,
            hide_index=True,
        )


def render_training_view() -> None:
    st.caption("CNN propia entrenada desde cero con HAM10000. No usa modelos preentrenados ni pesos externos.")
    metadata_path = find_ham10000_metadata()
    render_prediction_view()
    render_dataset_summary(metadata_path)
    render_training_results()


def render_clustering_view() -> None:
    st.caption("Analisis exploratorio no supervisado para agrupar imagenes por similitud visual.")

    with st.sidebar:
        st.header("Configuracion de clustering")
        method = st.selectbox("Metodo", ["kmeans", "dbscan"])
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
        return

    if len(uploaded_files) < 2:
        st.warning("Se necesitan al menos dos imagenes para evaluar clusters.")
        return

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


def main() -> None:
    st.set_page_config(page_title="Patrones de piel", layout="wide")

    st.title("Patrones de piel en dermatologia")
    st.caption("Entrenamiento propio con HAM10000 y analisis exploratorio de lesiones dermatologicas.")

    training_tab, clustering_tab = st.tabs(["Reconocimiento HAM10000", "Clustering exploratorio"])

    with training_tab:
        render_training_view()

    with clustering_tab:
        render_clustering_view()


if __name__ == "__main__":
    main()
