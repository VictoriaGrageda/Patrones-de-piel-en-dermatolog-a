import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from PIL import ImageOps

from src.skin_patterns.config import MODELS_DIR, PipelineConfig, RAW_DATA_DIR, REPORTS_DIR
from src.skin_patterns.ham10000 import SmallDermCnn
from src.skin_patterns.pipeline import build_feature_matrix


HAM10000_LABELS = {
    "akiec": "Queratosis actinica / enfermedad de Bowen",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesiones benignas tipo queratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanociticos",
    "vasc": "Lesiones vasculares",
}


# Lee archivos JSON generados por el entrenamiento supervisado HAM10000.
def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_ham10000_model(model_path: str) -> tuple[SmallDermCnn, dict[str, int], dict[int, str], int]:
    # Carga la CNN supervisada ya entrenada con etiquetas HAM10000.
    # Este modelo predice clases reales como mel, nv, bcc, etc.
    checkpoint = torch.load(model_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {int(key): value for key, value in checkpoint["index_to_label"].items()}
    image_size = int(checkpoint.get("config", {}).get("image_size", 128))

    model = SmallDermCnn(num_classes=len(label_to_index))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_to_index, index_to_label, image_size


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    # Prepara la imagen nueva igual que durante el entrenamiento de la CNN.
    # Se corrige orientacion, se redimensiona, se normaliza y se pasa a tensor.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def predict_ham10000(image: Image.Image) -> pd.DataFrame:
    # Flujo supervisado: usa la CNN entrenada con HAM10000 para clasificar
    # una imagen en una de las clases diagnosticas del dataset.
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


@st.cache_resource
def load_clustering_artifact(model_path: str) -> dict:
    # Carga el artefacto del aprendizaje no supervisado:
    # configuracion, scaler, PCA, modelo de clustering y metricas.
    return joblib.load(model_path)


def predict_dbscan_labels(model: object, embedding: np.ndarray) -> np.ndarray:
    # DBSCAN no tiene metodo predict. Para imagenes nuevas se asigna el
    # cluster del punto nucleo mas cercano; si queda lejos, se marca como -1.
    if not hasattr(model, "components_") or not hasattr(model, "core_sample_indices_"):
        return np.full(embedding.shape[0], -1, dtype=int)

    core_samples = model.components_
    if len(core_samples) == 0:
        return np.full(embedding.shape[0], -1, dtype=int)

    distances = np.linalg.norm(embedding[:, None, :] - core_samples[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    labels = model.labels_[model.core_sample_indices_][nearest]
    labels = np.asarray(labels, dtype=int)
    labels[distances.min(axis=1) > model.eps] = -1
    return labels


def predict_fuzzy_labels(model: dict, embedding: np.ndarray) -> np.ndarray:
    # Fuzzy C-Means guarda centros. La imagen nueva se asigna al centro mas cercano.
    centers = np.asarray(model["centers"])
    distances = np.linalg.norm(embedding[:, None, :] - centers[None, :, :], axis=2)
    return distances.argmin(axis=1)


def predict_cluster_labels(model: object, embedding: np.ndarray) -> np.ndarray:
    # Unifica la inferencia para K-Means, GMM, Fuzzy C-Means y DBSCAN.
    if hasattr(model, "predict"):
        return np.asarray(model.predict(embedding))
    if isinstance(model, dict) and "centers" in model:
        return predict_fuzzy_labels(model, embedding)
    return predict_dbscan_labels(model, embedding)


def project_uploaded_images(input_dir: Path, artifact: dict) -> pd.DataFrame:
    # Flujo no supervisado: las imagenes subidas no reentrenan el clustering.
    # Se extraen caracteristicas, se aplica el mismo scaler y PCA del entrenamiento,
    # y el modelo guardado decide en que cluster cae cada imagen.
    config = artifact.get("config") or PipelineConfig()
    names, matrix = build_feature_matrix(list(input_dir.iterdir()), config)
    scaled = artifact["scaler"].transform(matrix)
    embedding = artifact["reducer"].transform(scaled)
    labels = predict_cluster_labels(artifact["cluster_model"], embedding)

    return pd.DataFrame(
        {
            "image": names,
            "cluster": labels,
            "x": embedding[:, 0],
            "y": embedding[:, 1] if embedding.shape[1] > 1 else 0.0,
        }
    )


def find_ham10000_metadata() -> Path:
    # Busca el CSV de HAM10000 en la ubicacion esperada del proyecto.
    default_path = RAW_DATA_DIR / "HAM10000" / "HAM10000_metadata.csv"
    if default_path.exists():
        return default_path

    matches = sorted(RAW_DATA_DIR.rglob("HAM10000_metadata.csv"))
    return matches[0] if matches else default_path


def render_dataset_summary(metadata_path: Path) -> None:
    # Muestra un resumen del dataset usado por el entrenamiento supervisado.
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
    # Muestra los resultados guardados por train_ham10000.py:
    # curvas de entrenamiento, reporte por clase y matriz de confusion.
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
    # Vista supervisada: reconoce una imagen con la CNN HAM10000 ya entrenada.
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
    # Pestaña de aprendizaje supervisado: prediccion CNN + informacion del entrenamiento.
    st.caption("CNN propia entrenada desde cero con HAM10000. No usa modelos preentrenados ni pesos externos.")
    metadata_path = find_ham10000_metadata()
    render_prediction_view()
    render_dataset_summary(metadata_path)
    render_training_results()


def render_clustering_view() -> None:
    # Pestaña de aprendizaje no supervisado: usa el clustering entrenado con HAM10000.
    # El grafico X/Y muestra el espacio PCA y ubica las imagenes subidas sobre ese mapa.
    st.caption("Inferencia con el modelo de clustering entrenado y visualizacion de coordenadas X/Y en PCA.")

    model_path = MODELS_DIR / "skin_pattern_model.joblib"
    report_path = REPORTS_DIR / "clustering_results.csv"

    if not model_path.exists():
        st.info("Primero entrena el clustering para usar el modelo guardado en las imagenes subidas.")
        st.code("python train.py --input data/raw --method kmeans --clusters 4", language="bash")
        return

    artifact = load_clustering_artifact(str(model_path))
    saved_config = artifact.get("config") or PipelineConfig()
    saved_method = artifact.get("method", "clustering")

    with st.sidebar:
        st.header("Modelo de clustering")
        st.metric("Metodo entrenado", str(saved_method).upper())
        st.metric("Tamano de imagen", f"{saved_config.image_size[0]}x{saved_config.image_size[1]}")
        if hasattr(artifact.get("reducer"), "n_components_"):
            st.metric("Componentes PCA", int(artifact["reducer"].n_components_))

    uploaded_files = st.file_uploader(
        "Carga imagenes dermatologicas para asignarlas al clustering entrenado",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="clustering_upload",
    )
    uploaded_files = list(uploaded_files or [])

    if len(uploaded_files) == 0:
        st.info("Carga una o mas imagenes para proyectarlas en el grafico X/Y y asignarles cluster.")
        return

    st.caption(f"Imagenes listas para procesar: {len(uploaded_files)}")

    with TemporaryDirectory() as tmpdir:
        # Streamlit entrega archivos en memoria; se guardan temporalmente para reutilizar
        # el mismo pipeline de lectura y extraccion de caracteristicas.
        input_dir = Path(tmpdir)
        for uploaded_file in uploaded_files:
            target = input_dir / uploaded_file.name
            target.write_bytes(uploaded_file.getbuffer())

        with st.spinner("Procesando imagenes con el modelo de clustering entrenado..."):
            table = project_uploaded_images(input_dir, artifact)

        metric_cols = st.columns(3)
        metric_names = ["silhouette", "davies_bouldin", "calinski_harabasz"]
        for col, name in zip(metric_cols, metric_names):
            value = artifact.get("metrics", {}).get(name)
            col.metric(name.replace("_", " ").title(), "N/A" if value is None else f"{value:.3f}")

        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("Grafico X/Y del PCA")
            fig, ax = plt.subplots(figsize=(7, 5))

            if report_path.exists():
                # Puntos de entrenamiento: imagenes HAM10000 usadas para crear el clustering.
                trained_table = pd.read_csv(report_path)
                ax.scatter(
                    trained_table["pc1"],
                    trained_table["pc2"],
                    c=trained_table["cluster"],
                    cmap="tab10",
                    s=28,
                    alpha=0.25,
                    label="Entrenamiento",
                )

            # Puntos nuevos: imagenes subidas por el usuario proyectadas al mismo PCA.
            scatter = ax.scatter(
                table["x"],
                table["y"],
                c=table["cluster"],
                cmap="tab10",
                s=130,
                marker="X",
                edgecolors="black",
                linewidths=0.8,
                label="Subidas",
            )
            ax.set_xlabel("X (PC1)")
            ax.set_ylabel("Y (PC2)")
            ax.grid(alpha=0.2)
            ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
            st.pyplot(fig)

        with right:
            st.subheader("Resultados de inferencia")
            display_table = table.copy()
            display_table["x"] = display_table["x"].round(4)
            display_table["y"] = display_table["y"].round(4)
            st.dataframe(display_table, use_container_width=True, hide_index=True)

        st.subheader("Imagenes agrupadas")
        for cluster_id in sorted(pd.unique(table["cluster"])):
            title = "Ruido / sin cluster" if int(cluster_id) == -1 else f"Cluster {cluster_id}"
            st.markdown(f"**{title}**")
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
