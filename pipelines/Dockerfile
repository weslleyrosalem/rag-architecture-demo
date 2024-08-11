FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade odh-elyra==3.16.7 Flask kafka-python watchdog kfp==2.8.0

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

# Adjust permissions to allow write access for any user in the group '0'
RUN chown -R root:0 /opt/app-root/src \
    && chmod -R g+rwX /opt/app-root/src

# Switch back to the default user (any non-root user OpenShift will assign)
USER 1001

# Copy entrypoint script with appropriate permissions
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# Define entrypoint
ENTRYPOINT ["/entrypoint.sh"]