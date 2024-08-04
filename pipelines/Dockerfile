#FROM quay.io/modh/runtime-images@sha256:0a3c137492e66c5fac89786f90dbbfdfd3eecd8d9c0e4e890aeeb8f558cd731d
FROM quay.io/modh/runtime-images@sha256:1711dfaeb5fea393ebe09dc957fcf36720ff2859c3afc47fe4e2110b68bc918f

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --upgrade elyra Flask kafka-python watchdog
#RUN pip install --upgrade Flask kafka-python

# Create Elyra directories (metadata, runtime, and cache)
RUN mkdir -p /opt/app-root/src/.local/share/jupyter/metadata \
           /opt/app-root/src/.local/share/jupyter/runtime \
           /opt/app-root/src/.local/share/jupyter/cache

# Copy and import runtime configurations
COPY runtimes/odh_dsp.json /opt/app-root/src/.local/share/jupyter/metadata/runtimes/
RUN elyra-metadata import runtimes --directory /opt/app-root/src/.local/share/jupyter/metadata/runtimes

# Set Elyra environment variables
ENV ELYRA_METADATA_HOME=/opt/app-root/src/.local/share/jupyter/metadata
ENV ELYRA_RUNTIME_DIR=/opt/app-root/src/.local/share/jupyter/runtime
ENV ELYRA_COMPONENT_CATALOG_DIR=/opt/app-root/src/.local/share/jupyter/cache

# Switch to root to modify permissions
USER root

# Adjust permissions (consider more restrictive permissions in production)
RUN chmod -R 777 /opt/app-root/src/.local/share/jupyter

# Switch back to the default user (if necessary)
USER 1001

# Copy entrypoint script with appropriate permissions
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# Define entrypoint
ENTRYPOINT ["/entrypoint.sh"]