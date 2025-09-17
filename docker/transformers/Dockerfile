# SpectraVision Unified Transformer Dockerfile
# Uses build arguments to support different transformer versions with centralized requirements

ARG TRANSFORMERS_VERSION=4.49
ARG VLMEVALKIT_VERSION=4.49
FROM spectravision-base:latest

# Re-declare args after FROM
ARG TRANSFORMERS_VERSION
ARG VLMEVALKIT_VERSION

# Set environment variables
ENV TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}
ENV VLMEVALKIT_VERSION=${VLMEVALKIT_VERSION}

# Install common dependencies
COPY docker/requirements/common.txt /tmp/common.txt
RUN pip install -r /tmp/common.txt

# Install version-specific dependencies - use multi-stage approach
ARG TRANSFORMERS_VERSION
COPY docker/requirements/transformers-${TRANSFORMERS_VERSION}.txt /tmp/version-specific.txt
RUN pip install -r /tmp/version-specific.txt

# Setup VLMEvalKit specific version directory
ARG VLMEVALKIT_VERSION
COPY VLMEvalKit-${VLMEVALKIT_VERSION}/ /workspace/VLMEvalKit/
WORKDIR /workspace/VLMEvalKit

# Install VLMEvalKit in development mode
RUN pip install -e .

# Apply version-specific patches (if needed)
COPY patches/ /tmp/patches/
ARG TRANSFORMERS_VERSION
RUN if [ -f /tmp/patches/transformers-${TRANSFORMERS_VERSION}.patch ]; then \
        patch -p1 < /tmp/patches/transformers-${TRANSFORMERS_VERSION}.patch; \
    fi

# Set model environment variables based on version
ARG CONTAINER_MODELS=""
ENV CONTAINER_MODELS=${CONTAINER_MODELS}

# Default command
CMD ["/bin/bash"]
