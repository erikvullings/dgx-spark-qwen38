# Patched SGLang isolated from all existing lanes. CUDA/ARM base pinned by digest.
FROM lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
ARG SGLANG_REV=079d4046003daf757d8a2daa44a3bb8935174b5e
ENV SGLANG_BUILD_RUST_EXTS=none MAX_JOBS=2 NINJA_NUM_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
RUN git clone --filter=blob:none https://github.com/inclusionAI/sglang.git /opt/ling-sglang && \
    git -C /opt/ling-sglang checkout --detach "$SGLANG_REV" && \
    test "$(git -C /opt/ling-sglang rev-parse HEAD)" = "$SGLANG_REV" && \
    python3 -m pip install --no-cache-dir -e '/opt/ling-sglang/python[all]' && \
    python3 -m pip freeze > /opt/ling-runtime-packages.txt
COPY ling-launch.py ling-precompile.py /opt/ling/
ENTRYPOINT ["python3"]
CMD ["/opt/ling/ling-launch.py"]
