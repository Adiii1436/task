from sentence_transformers import SentenceTransformer

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save the model to the specified directory
model.save("local_models/all-MiniLM-L6-v2")