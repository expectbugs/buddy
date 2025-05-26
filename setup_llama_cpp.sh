#!/bin/bash

# llama.cpp Setup Script with CUDA Support
# Optimized for RTX 3090

set -e

echo "=== Setting up llama.cpp with CUDA support ==="

# Clone llama.cpp if not exists
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
else
    echo "llama.cpp directory exists, updating..."
    cd llama.cpp
    git pull
    cd ..
fi

cd llama.cpp

# Build with CUDA support using CMake
echo "Building llama.cpp with CUDA support using CMake..."
mkdir -p build
cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
cd ..

# Create models directory if not exists
if [ ! -d "models" ]; then
    mkdir -p models
fi

echo "=== Build complete ==="
echo ""
echo "IMPORTANT: Place your Hermes-2-Pro-Mistral-10.7B-Q6_K model file in:"
echo "$(pwd)/models/"
echo ""
echo "The model file should be named something like:"
echo "hermes-2-pro-mistral-10.7b.Q6_K.gguf"
echo ""
echo "You can download GGUF models from Hugging Face"

cd ..

echo "=== Creating Python bindings installation script ==="
cat > install_llama_cpp_python.sh << 'EOF'
#!/bin/bash
# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
EOF

chmod +x install_llama_cpp_python.sh

echo "=== Setup complete ==="
echo "Run ./install_llama_cpp_python.sh if you want to use Python bindings"