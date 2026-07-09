FROM python:3.10-slim

# Override with --build-arg APT_MIRROR=<host> for a faster build; defaults to public Debian mirrors.
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
        find /etc/apt -name '*.list' -o -name '*.sources' | xargs -r sed -i "s#deb.debian.org#${APT_MIRROR}#g; s#security.debian.org#${APT_MIRROR}#g"; \
    fi \
    && n=0; until [ "$n" -ge 5 ]; do \
        timeout 300 apt-get update && timeout 300 apt-get install -y --no-install-recommends ca-certificates libgomp1 && break; \
        n=$((n+1)); echo "apt-get attempt $n failed, retrying..."; sleep 5; \
    done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Override with --build-arg PIP_INDEX_URL=<mirror> for a faster build; defaults to public PyPI.
# PIP_EXTRA_INDEX_URL is an optional fallback (e.g. public PyPI) for packages missing from a mirror.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

COPY requirements.min.txt ./
RUN python -m pip install --no-cache-dir --timeout 60 --retries 30 -U pip setuptools wheel \
    && python -m pip install --no-cache-dir --timeout 60 --retries 30 -r requirements.min.txt

COPY requirements.native-cu126.txt ./
RUN python -m pip install --no-cache-dir --timeout 60 --retries 30 --no-deps -r requirements.native-cu126.txt

COPY oriongeno/ ./oriongeno/
COPY main.py LICENSE README.md ./

ENTRYPOINT ["python", "main.py"]
