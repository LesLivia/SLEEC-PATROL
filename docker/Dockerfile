# check=skip=FromPlatformFlagConstDisallowed
# The SLEEC-PATROL web app with everything it needs preinstalled. Build from the repository
# root, which holds the code it copies; docker/Dockerfile.dockerignore limits what is sent:
#   docker build -f docker/Dockerfile -t sleec-patrol .
#   docker run --rm -p 8000:8000 sleec-patrol      then open http://localhost:8000
#
# maraboupy 2.0.0 is published for x86-64 Linux only, so the image is always linux/amd64
# (Apple-silicon Macs run it under emulation). The base image is pinned by digest, so a
# rebuild starts from exactly the same official python:3.10-slim.
FROM --platform=linux/amd64 python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Only what the web app imports, at the versions in environment.yml; torch and tensorflow
# are for training and exporting models, which the web app never does.
RUN pip install --no-cache-dir --root-user-action=ignore \
        maraboupy==2.0.0 \
        numpy==1.26.4 \
        onnx==1.17.0 \
        onnxruntime==1.23.2 \
        textx==4.2.3 \
        arpeggio==2.0.3 \
        jinja2==3.1.6 \
        fastapi==0.143.0 \
        uvicorn==0.54.0 \
        python-multipart==0.0.32

# The code is copied last, so changing it rebuilds only these steps.
WORKDIR /app
COPY legos_sleec/sleec-grammar.txt legos_sleec/
COPY verifier/__init__.py verifier/sleec2marabou_parser.py verifier/
COPY webapp/ webapp/

EXPOSE 8000
# 0.0.0.0: inside a container, a server on localhost cannot be reached from outside it.
CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
