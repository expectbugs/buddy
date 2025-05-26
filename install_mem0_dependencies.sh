#!/bin/bash

# Native Installation Script for Mem0 with Qdrant and Neo4j Support
# Gentoo Linux with OpenRC - No Docker/Systemd

set -e

echo "=== Installing Dependencies Natively (Gentoo/OpenRC) ==="

# Check if running on Gentoo
if [ ! -f /etc/gentoo-release ]; then
    echo "Warning: This script is optimized for Gentoo Linux"
    echo "Continuing anyway..."
fi

# Update Portage
echo "Syncing Portage tree..."
if command -v emerge &> /dev/null; then
    sudo emerge --sync
else
    echo "Error: This script requires Gentoo Linux with Portage"
    exit 1
fi

# Check and install system dependencies via Portage if needed
echo "Checking system dependencies..."

# Check which packages need to be installed
PACKAGES_TO_INSTALL=""

if ! command -v wget &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL net-misc/wget"
fi

if ! command -v curl &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL net-misc/curl"
fi

if ! command -v unzip &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL app-arch/unzip"
fi

if ! command -v java &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL virtual/jdk"
fi

if ! command -v cmake &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL dev-build/cmake"
fi

if ! command -v git &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL dev-vcs/git"
fi

if ! command -v python3 &> /dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL dev-lang/python"
fi

if [ -n "$PACKAGES_TO_INSTALL" ]; then
    echo "Installing missing packages: $PACKAGES_TO_INSTALL"
    sudo emerge --quiet --noreplace $PACKAGES_TO_INSTALL
else
    echo "All required system packages are already installed"
fi

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install mem0ai qdrant-client neo4j python-dotenv pyyaml

# Install additional useful packages
pip install sentence-transformers torch transformers

echo "=== Installing Qdrant Natively ==="

# Create directories
mkdir -p ~/qdrant
cd ~/qdrant

# Download Qdrant binary
echo "Downloading Qdrant..."
QDRANT_VERSION="v1.7.4"
wget https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
chmod +x qdrant

# Create Qdrant configuration
cat > config.yaml << 'EOF'
log_level: INFO
storage:
  storage_path: ./storage
service:
  http_port: 6333
  grpc_port: 6334
  host: 0.0.0.0
  max_request_size_mb: 32
  max_workers: 0
  enable_cors: true
cluster:
  enabled: false
EOF

# Create Qdrant OpenRC init script
sudo tee /etc/init.d/qdrant > /dev/null << 'EOF'
#!/sbin/openrc-run

name="Qdrant Vector Database"
description="Qdrant Vector Database Service"

user="$(logname 2>/dev/null || echo $USER)"
pidfile="/var/run/qdrant.pid"
command="/home/${user}/qdrant/qdrant"
command_args="--config-path /home/${user}/qdrant/config.yaml"
command_background="yes"
command_user="${user}"
directory="/home/${user}/qdrant"

depend() {
    need net
    after logger
}

start_pre() {
    checkpath --directory --owner ${user}:${user} --mode 0755 \
        /var/run /var/log
}
EOF

sudo chmod +x /etc/init.d/qdrant

# Add Qdrant to default runlevel and start
sudo rc-update add qdrant default
sudo rc-service qdrant start

cd ~/buddy

echo "=== Installing Neo4j Natively ==="

# Create directories
mkdir -p ~/neo4j
cd ~/neo4j

# Download Neo4j Community Edition
echo "Downloading Neo4j..."
NEO4J_VERSION="5.15.0"
wget https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz
tar -xzf neo4j-community-${NEO4J_VERSION}-unix.tar.gz
mv neo4j-community-${NEO4J_VERSION} neo4j-community
cd neo4j-community

# Configure Neo4j
echo "Configuring Neo4j..."

# Update neo4j.conf
cat >> conf/neo4j.conf << 'EOF'

# Enable HTTP connector
server.http.enabled=true
server.http.listen_address=0.0.0.0:7474

# Enable Bolt connector
server.bolt.enabled=true
server.bolt.listen_address=0.0.0.0:7687

# Set initial password
server.databases.default_to_read_only=false
dbms.security.auth_enabled=true

# APOC configuration
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Memory settings
server.memory.heap.initial_size=1G
server.memory.heap.max_size=2G
server.memory.pagecache.size=1G
EOF

# Download and install APOC plugin
echo "Installing APOC plugin..."
APOC_VERSION="5.15.0"
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-core.jar -O plugins/apoc-${APOC_VERSION}-core.jar

# Set initial password
bin/neo4j-admin dbms set-initial-password password123

# Create Neo4j OpenRC init script
sudo tee /etc/init.d/neo4j > /dev/null << 'EOF'
#!/sbin/openrc-run

name="Neo4j Graph Database"
description="Neo4j Graph Database Service"

user="$(logname 2>/dev/null || echo $USER)"
pidfile="/home/${user}/neo4j/neo4j-community/run/neo4j.pid"
command="/home/${user}/neo4j/neo4j-community/bin/neo4j"
command_user="${user}"
directory="/home/${user}/neo4j/neo4j-community"

depend() {
    need net
    after logger
}

start() {
    ebegin "Starting Neo4j"
    start-stop-daemon --start --quiet \
        --pidfile "${pidfile}" \
        --user "${user}" \
        --chdir "${directory}" \
        --exec "${command}" -- start
    eend $?
}

stop() {
    ebegin "Stopping Neo4j"
    start-stop-daemon --stop --quiet \
        --pidfile "${pidfile}" \
        --user "${user}" \
        --chdir "${directory}" \
        --exec "${command}" -- stop
    eend $?
}

status() {
    if [ -f "${pidfile}" ]; then
        einfo "Neo4j is running"
        return 0
    else
        einfo "Neo4j is not running"
        return 1
    fi
}
EOF

sudo chmod +x /etc/init.d/neo4j

# Add Neo4j to default runlevel and start
sudo rc-update add neo4j default
sudo rc-service neo4j start

cd ~/buddy

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 15

echo "=== Checking service status ==="
if rc-service qdrant status &>/dev/null; then
    echo "✓ Qdrant is running"
else
    echo "✗ Qdrant failed to start"
    echo "Check status with: sudo rc-service qdrant status"
fi

if rc-service neo4j status &>/dev/null; then
    echo "✓ Neo4j is running"
else
    echo "✗ Neo4j failed to start"
    echo "Check status with: sudo rc-service neo4j status"
fi

echo "=== Installation Complete (Gentoo/OpenRC) ==="
echo "Qdrant dashboard: http://localhost:6333/dashboard"
echo "Neo4j browser: http://localhost:7474"
echo "Neo4j credentials: neo4j/password123"
echo ""
echo "OpenRC Service management:"
echo "  Start Qdrant: sudo rc-service qdrant start"
echo "  Stop Qdrant: sudo rc-service qdrant stop"
echo "  Status: sudo rc-service qdrant status"
echo ""
echo "  Start Neo4j: sudo rc-service neo4j start"
echo "  Stop Neo4j: sudo rc-service neo4j stop"
echo "  Status: sudo rc-service neo4j status"
echo ""
echo "  Add to runlevel: sudo rc-update add qdrant default"
echo "  Remove from runlevel: sudo rc-update del qdrant default"
echo ""
echo "Next steps:"
echo "1. Run ./setup_llama_cpp.sh to install llama.cpp with CUDA support"
echo "2. Configure your environment variables in .env file"
echo "3. Run ./launch_hermes_with_mem0.sh to start the AI model"