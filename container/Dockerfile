FROM python:3.11-slim-bookworm
# Resolve and record the resulting image digest before collecting published runs.
# No credentials or user files belong in this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils findutils grep sed gawk diffutils man-db manpages \
    info less file jq bc ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
CMD ["bash"]

