from flask import Flask, render_template, request, url_for
import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import numpy as np
from sklearn.decomposition import PCA

# Initialize Flask app
app = Flask(__name__)

# Load CLIP model and embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

EMBEDDINGS_FILE = "image_embeddings.pickle"
df = pd.read_pickle(EMBEDDINGS_FILE)

# Fit PCA model on the dataset embeddings
all_embeddings = np.vstack(df["embedding"].values)
pca_model = PCA()
pca_model.fit(all_embeddings)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query_type = request.form.get("query_type")
        text_query = request.form.get("text_query")
        image_file = request.files.get("image_query")
        use_pca = request.form.get("use_pca") == "on"
        k_components = int(request.form.get("k_principal_components", 50))
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5)) if query_type == "hybrid" else None

        # Initialize embeddings
        text_embedding = None
        image_embedding = None

        # Compute Text Embedding
        if text_query:
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            text_embedding = model.encode_text(tokenizer([text_query]).to(device))
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)

        # Compute Image Embedding
        if image_file:
            image = preprocess(Image.open(image_file).convert("RGB")).unsqueeze(0).to(device)
            image_embedding = model.encode_image(image)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)

        # Apply PCA if enabled
        if use_pca:
            if text_embedding is not None:
                text_embedding_np = text_embedding.detach().cpu().numpy()
                text_embedding_pca = pca_model.transform(text_embedding_np)[:, :k_components]
                text_embedding = torch.tensor(text_embedding_pca, device=device)
                text_embedding = F.normalize(text_embedding, p=2, dim=-1)

            if image_embedding is not None:
                image_embedding_np = image_embedding.detach().cpu().numpy()
                image_embedding_pca = pca_model.transform(image_embedding_np)[:, :k_components]
                image_embedding = torch.tensor(image_embedding_pca, device=device)
                image_embedding = F.normalize(image_embedding, p=2, dim=-1)

            # Apply PCA to the database embeddings
            database_embeddings_pca = pca_model.transform(all_embeddings)[:, :k_components]
            database_embeddings = torch.tensor(database_embeddings_pca, device=device)
            database_embeddings = F.normalize(database_embeddings, p=2, dim=-1)
        else:
            database_embeddings = torch.tensor(
                np.vstack(df["embedding"].values), device=device
            )

        # Combine embeddings for hybrid query
        if query_type == "hybrid" and text_embedding is not None and image_embedding is not None:
            query_embedding = F.normalize(
                hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding, p=2, dim=-1
            )
        elif query_type == "text" and text_embedding is not None:
            query_embedding = text_embedding
        elif query_type == "image" and image_embedding is not None:
            query_embedding = image_embedding
        else:
            query_embedding = None

        # Calculate cosine similarities and retrieve top 5 results
        if query_embedding is not None:
            cos_similarities = torch.matmul(query_embedding, database_embeddings.T).squeeze(0).tolist()
            top_indices = torch.topk(torch.tensor(cos_similarities), 5).indices.tolist()
            results = [
                {
                    "file_name": url_for('static', filename=f"coco_images_resized/{df.iloc[idx]['file_name']}"),
                    "similarity": cos_similarities[idx]
                }
                for idx in top_indices
            ]

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
