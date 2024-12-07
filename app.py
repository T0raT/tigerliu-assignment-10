from flask import Flask, render_template, request, url_for
import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from sklearn.decomposition import PCA
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained CLIP model and image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Load image embeddings
EMBEDDINGS_FILE = "image_embeddings.pickle"
IMAGES_FOLDER = "static/coco_images_resized"  # Make sure images are under the static folder
df = pd.read_pickle(EMBEDDINGS_FILE)

# Perform PCA on the embeddings (first 10,000 for efficiency)
all_embeddings = np.vstack(df["embedding"].values)[:10000]
pca_model = PCA()
pca_model.fit(all_embeddings)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query_type = request.form.get("query_type")
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))
        use_pca = request.form.get("use_pca") == "on"
        k_principal_components = int(request.form.get("k_principal_components", 50))

        # Handle text query
        if query_type == "text" or query_type == "hybrid":
            text_query = request.form.get("text_query")
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text_embedding = F.normalize(model.encode_text(tokenizer([text_query])).to(device), p=2, dim=-1)

        # Handle image query
        if query_type == "image" or query_type == "hybrid":
            image_file = request.files.get("image_query")
            if image_file:
                image = preprocess(Image.open(image_file).convert("RGB")).unsqueeze(0).to(device)
                image_embedding = model.encode_image(image).detach().cpu().numpy()

                # Apply PCA when selected
                if use_pca:
                    image_embedding = pca_model.transform(image_embedding)[:, :k_principal_components]

                image_embedding = torch.tensor(image_embedding, device=device)
                image_embedding = F.normalize(image_embedding, p=2, dim=-1)

        # Combine image and text embeddings if hybrid
        if query_type == "hybrid" and image_file and text_query:
            query_embedding = F.normalize(hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding, p=2, dim=-1)
        elif query_type == "text":
            query_embedding = text_embedding
        elif query_type == "image":
            query_embedding = image_embedding

        # Compute cosine similarities
        if query_embedding is not None:
            if use_pca:
                # Apply PCA to database embeddings
                database_embeddings = np.vstack(df["embedding"].values)
                reduced_embeddings = pca_model.transform(database_embeddings)[:, :k_principal_components]
                reduced_embeddings = torch.tensor(reduced_embeddings, device=device)
            else:
                # Use original embeddings
                reduced_embeddings = torch.tensor(np.vstack(df["embedding"].values), device=device)

            # Normalize database embeddings
            reduced_embeddings = F.normalize(reduced_embeddings, p=2, dim=-1)

            # Calculate similarities
            cos_similarities = torch.matmul(query_embedding, reduced_embeddings.T).squeeze(0).tolist()

        # Retrieve top 5 results
        top_indices = torch.topk(torch.tensor(cos_similarities), 5).indices
        top_indices = top_indices.tolist()  # Convert tensor to list of integers
        results = [{"file_name": os.path.join(IMAGES_FOLDER, df.iloc[idx]["file_name"]), "similarity": cos_similarities[idx]} for idx in top_indices]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)

