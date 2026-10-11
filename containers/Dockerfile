# SPDX-License-Identifier: Unlicense
FROM python:3.12-alpine@sha256:1b668429b3511ab407d8e00648891631b0b1a4d7e15e3ca70f38ab5b91ad4ab4
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN addgroup -g 1000 tenant && adduser -D -u 1000 -G tenant tenant
WORKDIR /app
COPY gpuctl.py /app/gpuctl.py
USER 1000:1000
ENTRYPOINT ["python", "/app/gpuctl.py"]
