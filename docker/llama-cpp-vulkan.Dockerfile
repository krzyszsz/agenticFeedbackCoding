FROM ubuntu:24.04 AS build

ARG LLAMA_CPP_REF=master
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake ninja-build pkg-config \
    python3 python3-venv python3-pip \
    libvulkan-dev vulkan-tools glslang-tools glslc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN set -eux; \
    for attempt in 1 2 3; do \
      rm -rf /opt/llama.cpp; \
      if git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp; then \
        break; \
      fi; \
      if [ "$attempt" -eq 3 ]; then \
        exit 1; \
      fi; \
      sleep 5; \
    done
WORKDIR /opt/llama.cpp
RUN git checkout ${LLAMA_CPP_REF}

RUN cmake -S . -B build -G Ninja \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target llama-server llama-cli

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libstdc++6 libgomp1 \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /opt/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=build /opt/llama.cpp/build/bin/libggml*.so* /usr/local/lib/
COPY --from=build /opt/llama.cpp/build/bin/libllama*.so* /usr/local/lib/
COPY --from=build /opt/llama.cpp/build/bin/libmtmd*.so* /usr/local/lib/
COPY docker/llama-cpp-run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/run.sh

ENV LD_LIBRARY_PATH=/usr/local/lib
EXPOSE 8161
ENTRYPOINT ["/usr/local/bin/run.sh"]
