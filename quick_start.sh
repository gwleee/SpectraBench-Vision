#!/bin/bash

# SpectraBench-Vision Quick Start Script
# AI Platform Team at KISTI Large-scale AI Research Center
# Supports both Docker (recommended) and local installation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "  SpectraBench-Vision Quick Start"
    echo "  Hardware-aware multimodal model evaluation system"
    echo "=================================================================="
    echo -e "${NC}"
    echo "Developed by AI Platform Team at KISTI Large-scale AI Research Center"
    echo ""
}

# Print step header
print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
    echo "----------------------------------------"
}

# Print info message
print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Print success message
print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_step "1" "Checking Prerequisites"
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3.9+ is required but not found"
        exit 1
    fi
    
    # Check pip
    if command_exists pip3 || command_exists pip; then
        print_success "pip found"
    else
        print_error "pip is required but not found"
        exit 1
    fi
    
    # Check git
    if command_exists git; then
        print_success "Git found"
    else
        print_error "Git is required but not found"
        exit 1
    fi
    
    # Check NVIDIA GPU
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "NVIDIA GPU detected: $GPU_INFO"
    else
        print_warning "nvidia-smi not found. CUDA-capable GPU recommended"
    fi
    
    echo ""
}

# Setup environment
setup_environment() {
    print_step "2" "Setting Up Environment"
    
    # Check if in virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Virtual environment active: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment detected"
        read -p "Create a new virtual environment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 -m venv spectrabench_env
            source spectrabench_env/bin/activate
            print_success "Virtual environment created and activated"
        fi
    fi
    
    echo ""
}

# Install dependencies
install_dependencies() {
    print_step "3" "Installing Dependencies"
    
    print_info "Installing core requirements..."
    pip install -r requirements.txt
    
    print_info "Running automated setup script..."
    python scripts/setup_dependencies.py
    
    print_success "Dependencies installed successfully"
    echo ""
}

# Run quick test
run_quick_test() {
    print_step "4" "Running Quick Test"
    
    print_info "Running availability tests to verify setup..."
    python scripts/main.py --mode test --test-time-limit 60
    
    print_success "Quick test completed"
    echo ""
}

# Show usage examples
show_usage() {
    print_step "5" "Usage Examples"
    
    echo "Now you can use SpectraBench-Vision with the following commands:"
    echo ""
    echo -e "${YELLOW}Interactive Mode:${NC}"
    echo "  python scripts/main.py"
    echo ""
    echo -e "${YELLOW}Availability Tests:${NC}"
    echo "  python scripts/main.py --mode test --test-time-limit 90"
    echo ""
    echo -e "${YELLOW}Full Evaluation:${NC}"
    echo "  python scripts/main.py --mode full --enable-monitoring"
    echo ""
    echo -e "${YELLOW}Specific Model-Benchmark:${NC}"
    echo "  python scripts/main.py --models \"InternVL2-2B\" --benchmarks \"MMBench\""
    echo ""
    echo -e "${YELLOW}Custom Hardware:${NC}"
    echo "  python scripts/main.py --hardware a100_single"
    echo ""
    
    print_info "For more options: python scripts/main.py --help"
    print_info "Documentation: README.md and SETUP.md"
    echo ""
}

# Docker setup
setup_docker() {
    print_step "1" "Docker Setup (Recommended)"
    
    # Check Docker
    if command_exists docker; then
        print_success "Docker found"
        
        # Check if Docker is running
        if docker info >/dev/null 2>&1; then
            print_success "Docker is running"
        else
            print_error "Docker is installed but not running. Please start Docker first."
            exit 1
        fi
    else
        print_error "Docker is required but not found"
        echo -e "${YELLOW}Please install Docker first:${NC}"
        echo "  • Ubuntu: sudo apt install docker.io"
        echo "  • macOS: brew install --cask docker"
        echo "  • Windows: Install Docker Desktop"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if command_exists nvidia-smi; then
        if docker info 2>/dev/null | grep -q "nvidia"; then
            print_success "NVIDIA Docker runtime detected"
        else
            print_warning "NVIDIA Docker runtime not detected. GPU support may not work."
            print_info "Install nvidia-container-toolkit for GPU support"
        fi
    else
        print_warning "nvidia-smi not found. CPU-only mode will be used."
    fi
    
    echo ""
}

# Start integrated Docker container
start_docker_container() {
    print_step "2" "Starting SpectraBench-Vision Container"
    
    print_info "Pulling latest integrated container..."
    if docker pull ghcr.io/gwleee/spectrabench-vision:latest; then
        print_success "Container image downloaded successfully"
    else
        print_error "Failed to download container image"
        print_info "Please check your internet connection and try again"
        exit 1
    fi
    
    print_info "Starting interactive container..."
    echo -e "${GREEN}Container started! Use 'python scripts/main.py' to begin evaluation.${NC}"
    echo -e "${GREEN}Your results will be saved to ./outputs/ on the host machine.${NC}"
    echo ""
    
    # Create outputs directory if it doesn't exist
    mkdir -p outputs
    
    # Start container
    docker run -it --rm \
        $(command_exists nvidia-smi && echo "--gpus all" || echo "") \
        -v "$(pwd)/outputs:/workspace/outputs" \
        -w /workspace \
        ghcr.io/gwleee/spectrabench-vision:latest
}

# Main execution
main() {
    print_banner
    
    # Parse command line arguments
    SKIP_PREREQUISITES=false
    SKIP_SETUP=false
    QUICK_MODE=false
    DOCKER_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-prerequisites)
                SKIP_PREREQUISITES=true
                shift
                ;;
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --docker)
                DOCKER_MODE=true
                shift
                ;;
            --help|-h)
                echo "Usage: ./quick_start.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --docker              Use Docker setup (recommended, fast)"
                echo "  --skip-prerequisites  Skip prerequisite checks (local mode)"
                echo "  --skip-setup         Skip environment setup"
                echo "  --quick              Quick mode (skip tests)"
                echo "  --help, -h           Show this help message"
                exit 0
                ;;
            *)
                print_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Execute steps based on mode
    if [[ "$DOCKER_MODE" == true ]]; then
        # Docker mode (recommended)
        setup_docker
        start_docker_container
    else
        # Local installation mode
        print_info "Using local installation mode. For faster setup, try: ./quick_start.sh --docker"
        echo ""
        
        if [[ "$SKIP_PREREQUISITES" != true ]]; then
            check_prerequisites
        fi
        
        if [[ "$SKIP_SETUP" != true ]]; then
            setup_environment
        fi
        
        install_dependencies
        
        if [[ "$QUICK_MODE" != true ]]; then
            run_quick_test
        fi
        
        show_usage
        
        print_success "SpectraBench-Vision setup completed successfully!"
        echo ""
    fi
    echo -e "${BLUE}Happy evaluating!${NC}"
}

# Run main function with all arguments
main "$@"