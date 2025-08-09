# 🔍 Multimodal AI Search Engine

A sophisticated image search engine that enables both text-to-image and image-to-image similarity search using state-of-the-art deep learning models. Built with CLIP (Contrastive Language-Image Pre-training) and FAISS for efficient vector search.

## 🌟 Features

- **🔤 Text-to-Image Search**: Find images using natural language descriptions
- **🖼️ Image-to-Image Search**: Upload an image to find visually similar ones  
- **⚡ Fast Search**: Sub-second query response times using FAISS indexing
- **🎯 High Accuracy**: Powered by OpenAI's CLIP-ViT-B-32 model
- **🎨 Modern UI**: Clean, responsive Gradio interface
- **🚀 GPU Accelerated**: CUDA support for faster inference

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Query    │    │   Image Query    │    │   CLIP Model    │
│   "red car"     │────▶   [Image.jpg]    │────▶  ViT-B-32      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   Embeddings    │
                                               │   512-dim       │
                                               └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │  FAISS Index    │
                                               │ Cosine Similarity│
                                               └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ Top-K Results   │
                                               │ + Scores        │
                                               └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aswnrj/multimodal-ai-search-engine.git
cd multimodal-search-engine
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download and prepare the dataset**
```bash
python download_images_hf.py
python embed_images_clip.py
python build_faiss_index.py
```

4. **Launch the application**
```bash
python app.py
```

The app will be available at `http://localhost:7860`

## 📊 Dataset

- **Source**: Caltech101 dataset via HuggingFace (`flwrlabs/caltech101`)
- **Size**: 500 randomly sampled images
- **Categories**: 101 different object classes
- **Format**: RGB images, various resolutions
- **Preprocessing**: Normalized for CLIP compatibility

## 🔧 Technical Implementation

### Core Components

1. **Image Embedding** (`embed_images_clip.py`)
   - Loads images and converts to RGB
   - Generates 512-dim embeddings using CLIP-ViT-B-32
   - Applies L2 normalization for cosine similarity

2. **FAISS Indexing** (`build_faiss_index.py`)
   - Creates IndexFlatIP for inner product search
   - Optimized for normalized embeddings
   - Persistent storage for fast loading

3. **Search Interface** (`search_image_and_text.py`)
   - Unified search functions for text and image queries
   - Real-time embedding generation
   - Efficient similarity computation

4. **Web Interface** (`app.py`)
   - Modern Gradio-based UI
   - Dual-mode search (text/image)
   - Result visualization with similarity scores

### Key Algorithms

- **CLIP**: Contrastive learning between text and images
- **Vision Transformer**: Self-attention mechanism for image processing
- **Cosine Similarity**: Semantic similarity measurement
- **FAISS IndexFlatIP**: Exhaustive search with inner product

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Search Latency | <100ms |
| Embedding Dimension | 512 |
| Index Size | ~1MB (500 images) |
| Memory Usage | ~2GB (with model) |
| GPU Utilization | ~1GB VRAM |

## 🛠️ Project Structure

```
multimodal-search-engine/
├── dataset/
│   ├── images/              # Downloaded images
│   ├── image_embeddings.npy # Precomputed embeddings
│   ├── image_filenames.npy  # Filename mappings
│   └── faiss_index.bin      # FAISS search index
├── download_images_hf.py    # Dataset downloader
├── embed_images_clip.py     # Embedding generator
├── build_faiss_index.py     # Index builder
├── search_image_and_text.py # Search utilities
├── app.py                   # Gradio web interface
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

## 🎯 Use Cases

- **E-commerce**: Product similarity search
- **Digital Asset Management**: Content organization
- **Research**: Dataset exploration and analysis  
- **Education**: Visual learning and discovery
- **Creative**: Inspiration and mood board creation

## 🔮 Future Enhancements

- [ ] Support for larger datasets (10K+ images)
- [ ] Multiple CLIP model variants comparison
- [ ] Advanced filtering options (category, color, etc.)
- [ ] Batch upload for multiple query images
- [ ] API endpoints for programmatic access
- [ ] Similarity threshold tuning
- [ ] Export search results functionality

## 📚 Dependencies

### Core ML Libraries
- **PyTorch**: Deep learning framework
- **Sentence Transformers**: CLIP model interface
- **FAISS**: Vector similarity search
- **NumPy**: Numerical computations

### Data & UI
- **Pillow**: Image processing
- **Gradio**: Web interface
- **Datasets**: HuggingFace dataset loading
- **TQDM**: Progress tracking

## 🏆 Academic Applications

This project demonstrates several key concepts valuable for AI/ML academic programs:

- **Multimodal Learning**: Cross-modal understanding between text and images
- **Vector Databases**: Efficient similarity search at scale
- **Transfer Learning**: Leveraging pre-trained models
- **UI/UX Design**: Making AI accessible through intuitive interfaces
- **System Architecture**: Building end-to-end ML pipelines

## 📖 References

1. Radford, A., et al. "Learning Transferable Visual Representations from Natural Language Supervision." ICML 2021.
2. Johnson, J., Douze, M., Jégou, H. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data 2019.
3. Fei-Fei, L., Fergus, R., Perona, P. "Learning generative visual models from few training examples." CVPR 2004.

## 📄 License

MIT License - feel free to use this project for educational and research purposes.

## 👨‍💻 Author

Aswin Raj Rajan

---
