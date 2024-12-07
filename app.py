from flask import Flask, render_template, request, url_for
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained CLIP model and image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Load image embeddings
EMBEDDINGS_FILE = "image_embeddings.pickle"
IMAGES_FOLDER = "static/coco_images_resized"
df = pd.read_pickle(EMBEDDINGS_FILE)


# Cosine similarity calculation
def cosine_similarity(query_embedding, database_embeddings):
    return torch.matmul(query_embedding, database_embeddings.T).squeeze(0)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query_type = request.form.get("query_type")

        if query_type == "text":
            # Handle text query
            text_query = request.form.get("text_query")
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            query_embedding = model.encode_text(tokenizer([text_query])).to(device)
        
        elif query_type == "image":
            # Handle image query
            image_file = request.files.get("image_query")
            if image_file:
                image = preprocess(Image.open(image_file).convert("RGB")).unsqueeze(0).to(device)
                query_embedding = model.encode_image(image)
        
        # Normalize query embedding
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)

        # Normalize the database embeddings
        database_embeddings = torch.tensor(
            np.vstack(df["embedding"].values), device=device
        )
        database_embeddings = F.normalize(database_embeddings, p=2, dim=-1)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, database_embeddings)

        # Get top 5 results
        top_indices = torch.topk(similarities, 5).indices.tolist()
        results = [{"file_name": os.path.join(IMAGES_FOLDER, df.iloc[idx]["file_name"]), 
                    "similarity": similarities[idx].item()} for idx in top_indices]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
