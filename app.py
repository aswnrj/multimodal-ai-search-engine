import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import os
from typing import List, Tuple, Optional
import time

# Configuration
META_PATH = "dataset/image_filenames.npy"
INDEX_PATH = "dataset/faiss_index.bin"
IMG_DIR = "dataset/images"

class MultimodalSearchEngine:
    def __init__(self):
        """Initialize the search engine with pre-built index and model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔍 Using device: {self.device}")
        
        # Load pre-built index and metadata
        self.index = faiss.read_index(INDEX_PATH)
        self.image_files = np.load(META_PATH)
        
        # Load CLIP model
        self.model = SentenceTransformer("clip-ViT-B-32", device=self.device)
        
        print(f"✅ Loaded index with {self.index.ntotal} images")
    
    def search_by_text(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for images matching a text query."""
        if not query.strip():
            return []
        
        start_time = time.time()
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(query_emb, k)
        search_time = time.time() - start_time
        
        results = []
        for j, i in enumerate(idxs[0]):
            if i != -1:  # Valid index
                img_path = os.path.join(IMG_DIR, self.image_files[i])
                results.append((img_path, float(scores[0][j]), search_time))
        
        return results
    
    def search_by_image(self, image: Image.Image, k: int = 5) -> List[Tuple[str, float]]:
        """Search for images visually similar to the given image."""
        if image is None:
            return []
        
        start_time = time.time()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        query_emb = self.model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
        query_emb = np.expand_dims(query_emb, axis=0)
        scores, idxs = self.index.search(query_emb, k)
        search_time = time.time() - start_time
        
        results = []
        for j, i in enumerate(idxs[0]):
            if i != -1:  # Valid index
                img_path = os.path.join(IMG_DIR, self.image_files[i])
                results.append((img_path, float(scores[0][j]), search_time))
        
        return results

# Initialize the search engine
try:
    search_engine = MultimodalSearchEngine()
    ENGINE_LOADED = True
except Exception as e:
    print(f"❌ Error loading search engine: {e}")
    ENGINE_LOADED = False

def format_results(results: List[Tuple[str, float, float]]) -> Tuple[List[str], str]:
    """Format search results for Gradio display."""
    if not results:
        return [], "No results found."
    
    image_paths = [result[0] for result in results]
    search_time = results[0][2] if results else 0
    
    # Create detailed results text
    results_text = f"🔍 **Search Results** (Search time: {search_time:.3f}s)\n\n"
    for i, (path, score, _) in enumerate(results, 1):
        filename = os.path.basename(path)
        # Extract label from filename (format: 00000_label.jpg)
        label = filename.split('_', 1)[1].rsplit('.', 1)[0] if '_' in filename else 'unknown'
        results_text += f"**{i}.** {label} (similarity: {score:.3f})\n"
    
    return image_paths, results_text

def text_search_interface(query: str, num_results: int) -> Tuple[List[str], str]:
    """Interface function for text-based search."""
    if not ENGINE_LOADED:
        return [], "❌ Search engine not loaded. Please check if all files are available."
    
    if not query.strip():
        return [], "Please enter a search query."
    
    try:
        results = search_engine.search_by_text(query, k=num_results)
        return format_results(results)
    except Exception as e:
        return [], f"❌ Error during search: {str(e)}"

def image_search_interface(image: Image.Image, num_results: int) -> Tuple[List[str], str]:
    """Interface function for image-based search."""
    if not ENGINE_LOADED:
        return [], "❌ Search engine not loaded. Please check if all files are available."
    
    if image is None:
        return [], "Please upload an image."
    
    try:
        results = search_engine.search_by_image(image, k=num_results)
        return format_results(results)
    except Exception as e:
        return [], f"❌ Error during search: {str(e)}"

def get_random_examples() -> List[str]:
    """Get random example queries."""
    examples = [
        "a cat sitting on a chair",
        "airplane in the sky",
        "red car on the road",
        "person playing guitar",
        "dog running in the park",
        "beautiful sunset landscape",
        "computer on a desk",
        "flowers in a garden"
    ]
    return examples

# Create the Gradio interface
with gr.Blocks(
    title="🔍 Multimodal AI Search Engine",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .gallery img {
        border-radius: 8px;
    }
    """
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>🔍 Multimodal AI Search Engine</h1>
        <p style="font-size: 18px; color: #666;">
            Search through 500 Caltech101 images using text descriptions or image similarity
        </p>
        <p style="font-size: 14px; color: #888;">
            Powered by CLIP-ViT-B-32 and FAISS for fast similarity search
        </p>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        
        # Text Search Tab
        with gr.Tab("📝 Text Search", id="text_search"):
            gr.Markdown("### Search images using natural language descriptions")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Describe what you're looking for (e.g., 'a red car', 'person with guitar')",
                        lines=2
                    )
                    
                with gr.Column(scale=1):
                    text_num_results = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Number of Results"
                    )
            
            text_search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
            
            # Examples
            gr.Examples(
                examples=get_random_examples()[:4],
                inputs=text_query,
                label="Example Queries"
            )
            
            with gr.Row():
                text_results = gr.Gallery(
                    label="Search Results",
                    show_label=True,
                    elem_id="text_gallery",
                    columns=5,
                    rows=1,
                    height="auto",
                    object_fit="contain"
                )
                text_info = gr.Markdown(label="Details")
        
        # Image Search Tab
        with gr.Tab("🖼️ Image Search", id="image_search"):
            gr.Markdown("### Find visually similar images")
            
            with gr.Row():
                with gr.Column(scale=2):
                    image_query = gr.Image(
                        label="Upload Query Image",
                        type="pil",
                        height=300
                    )
                    
                with gr.Column(scale=1):
                    image_num_results = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Number of Results"
                    )
            
            image_search_btn = gr.Button("🔍 Search Similar", variant="primary", size="lg")
            
            with gr.Row():
                image_results = gr.Gallery(
                    label="Similar Images",
                    show_label=True,
                    elem_id="image_gallery",
                    columns=5,
                    rows=1,
                    height="auto",
                    object_fit="contain"
                )
                image_info = gr.Markdown(label="Details")
        
        # About Tab
        with gr.Tab("ℹ️ About", id="about"):
            gr.Markdown("""
            ### 🔬 Technical Details
            
            This multimodal search engine demonstrates advanced AI techniques for content-based image retrieval:
            
            **🧠 Model Architecture:**
            - **CLIP-ViT-B-32**: OpenAI's Contrastive Language-Image Pre-training model
            - **Vision Transformer**: Processes images using attention mechanisms
            - **Dual-encoder**: Separate encoders for text and images mapping to shared embedding space
            
            **⚡ Search Infrastructure:**
            - **FAISS**: Facebook AI Similarity Search for efficient vector operations
            - **Cosine Similarity**: Measures semantic similarity in embedding space
            - **Inner Product Index**: Optimized for normalized embeddings
            
            **📊 Dataset:**
            - **Caltech101**: 500 randomly sampled images from 101 object categories
            - **Preprocessing**: RGB conversion, CLIP-compatible normalization
            - **Embeddings**: 512-dimensional feature vectors per image
            
            **🚀 Performance Features:**
            - **GPU Acceleration**: CUDA support for faster inference
            - **Batch Processing**: Efficient embedding computation
            - **Real-time Search**: Sub-second query response times
            - **Normalized Embeddings**: L2 normalization for consistent similarity scores
            
            **🎯 Applications:**
            - Content-based image retrieval
            - Visual search engines
            - Cross-modal similarity matching
            - Dataset exploration and analysis
            
            ### 🛠️ Implementation Highlights
            - Modular architecture with separate indexing and search components
            - Error handling and graceful degradation
            - Configurable result counts and similarity thresholds
            - Professional UI with responsive design
            """)
    
    # Event handlers
    text_search_btn.click(
        fn=text_search_interface,
        inputs=[text_query, text_num_results],
        outputs=[text_results, text_info]
    )
    
    image_search_btn.click(
        fn=image_search_interface,
        inputs=[image_query, image_num_results],
        outputs=[image_results, image_info]
    )
    
    # Auto-search on Enter key for text
    text_query.submit(
        fn=text_search_interface,
        inputs=[text_query, text_num_results],
        outputs=[text_results, text_info]
    )

# Launch configuration
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Multimodal AI Search Engine")
    print("="*50)
    
    if ENGINE_LOADED:
        print(f"✅ Search engine ready with {search_engine.index.ntotal} images")
        print(f"✅ Using device: {search_engine.device}")
    else:
        print("❌ Search engine failed to load")
    
    print("\n💡 Usage Tips:")
    print("- Text search: Use natural language descriptions")
    print("- Image search: Upload any image to find similar ones")
    print("- Adjust result count using the slider")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )