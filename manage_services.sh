#!/bin/bash

# Service Management Script for Native Qdrant and Neo4j (Gentoo/OpenRC)
# This script helps manage the Qdrant and Neo4j services using OpenRC

set -e

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Function to show service status
show_status() {
    print_header "Service Status (OpenRC)"
    
    echo "Qdrant Service:"
    if rc-service qdrant status &>/dev/null; then
        print_success "Running"
        echo "  Dashboard: http://localhost:6333/dashboard"
        echo "  Runlevel: $(rc-update show | grep qdrant | awk '{print $2}' || echo 'Not in any runlevel')"
    else
        print_error "Not running"
        echo "  Start with: sudo rc-service qdrant start"
    fi
    
    echo ""
    echo "Neo4j Service:"
    if rc-service neo4j status &>/dev/null; then
        print_success "Running"
        echo "  Browser: http://localhost:7474"
        echo "  Credentials: neo4j/password123"
        echo "  Runlevel: $(rc-update show | grep neo4j | awk '{print $2}' || echo 'Not in any runlevel')"
    else
        print_error "Not running"
        echo "  Start with: sudo rc-service neo4j start"
    fi
}

# Function to start services
start_services() {
    print_header "Starting Services (OpenRC)"
    
    echo "Starting Qdrant..."
    if sudo rc-service qdrant start; then
        print_success "Qdrant started"
    else
        print_error "Failed to start Qdrant"
        echo "Check logs with: sudo rc-service qdrant status"
    fi
    
    echo "Starting Neo4j..."
    if sudo rc-service neo4j start; then
        print_success "Neo4j started"
    else
        print_error "Failed to start Neo4j"
        echo "Check logs with: sudo rc-service neo4j status"
    fi
    
    echo ""
    print_warning "Waiting for services to initialize..."
    sleep 10
    show_status
}

# Function to stop services
stop_services() {
    print_header "Stopping Services (OpenRC)"
    
    echo "Stopping Qdrant..."
    if sudo rc-service qdrant stop; then
        print_success "Qdrant stopped"
    else
        print_error "Failed to stop Qdrant"
    fi
    
    echo "Stopping Neo4j..."
    if sudo rc-service neo4j stop; then
        print_success "Neo4j stopped"
    else
        print_error "Failed to stop Neo4j"
    fi
}

# Function to restart services
restart_services() {
    print_header "Restarting Services"
    stop_services
    echo ""
    start_services
}

# Function to show logs
show_logs() {
    print_header "Service Logs (OpenRC)"
    
    echo "Choose service to view logs:"
    echo "1) Qdrant"
    echo "2) Neo4j"
    echo "3) Both"
    echo "4) OpenRC service status"
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            echo "Qdrant service status and output:"
            sudo rc-service qdrant status
            echo ""
            echo "Qdrant log file (if available):"
            if [ -f ~/qdrant/qdrant.log ]; then
                tail -50 ~/qdrant/qdrant.log
            else
                echo "No log file found at ~/qdrant/qdrant.log"
            fi
            ;;
        2)
            echo "Neo4j service status and output:"
            sudo rc-service neo4j status
            echo ""
            echo "Neo4j logs:"
            if [ -d ~/neo4j/neo4j-community/logs ]; then
                echo "Debug log (last 25 lines):"
                tail -25 ~/neo4j/neo4j-community/logs/debug.log 2>/dev/null || echo "No debug.log found"
                echo ""
                echo "Neo4j log (last 25 lines):"
                tail -25 ~/neo4j/neo4j-community/logs/neo4j.log 2>/dev/null || echo "No neo4j.log found"
            else
                echo "No log directory found"
            fi
            ;;
        3)
            echo "=== Qdrant ==="
            sudo rc-service qdrant status
            echo ""
            echo "=== Neo4j ==="
            sudo rc-service neo4j status
            ;;
        4)
            echo "OpenRC service status:"
            echo "All services in default runlevel:"
            rc-update show default | grep -E "(qdrant|neo4j)" || echo "No services found in default runlevel"
            echo ""
            echo "Current service states:"
            sudo rc-status
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# Function to test connections
test_connections() {
    print_header "Testing Connections"
    
    echo "Testing Qdrant API..."
    if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
        print_success "Qdrant API is accessible"
    else
        print_error "Cannot connect to Qdrant API"
    fi
    
    echo "Testing Neo4j web interface..."
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        print_success "Neo4j web interface is accessible"
    else
        print_error "Cannot connect to Neo4j web interface"
    fi
    
    echo "Testing Neo4j bolt connection..."
    if timeout 5 bash -c "</dev/tcp/localhost/7687" 2>/dev/null; then
        print_success "Neo4j bolt port is accessible"
    else
        print_error "Cannot connect to Neo4j bolt port"
    fi
}

# Function to show system resources
show_resources() {
    print_header "System Resources"
    
    echo "Memory usage:"
    free -h
    echo ""
    
    echo "Disk usage (relevant directories):"
    du -sh ~/qdrant 2>/dev/null || echo "Qdrant directory not found"
    du -sh ~/neo4j 2>/dev/null || echo "Neo4j directory not found"
    echo ""
    
    echo "Service resource usage:"
    if rc-service qdrant status &>/dev/null; then
        echo "Qdrant process:"
        ps aux | grep -v grep | grep qdrant || echo "Process not found"
    fi
    
    if rc-service neo4j status &>/dev/null; then
        echo "Neo4j processes:"
        ps aux | grep -v grep | grep neo4j | head -5 || echo "Process not found"
    fi
    
    echo ""
    echo "OpenRC runlevel information:"
    echo "Current runlevel: $(rc-status --runlevel)"
    echo "Services in current runlevel:"
    rc-status | grep -E "(qdrant|neo4j)" || echo "No qdrant/neo4j services found in current runlevel"
}

# Main menu
show_menu() {
    echo ""
    print_header "Service Management Menu (OpenRC)"
    echo "1) Show status"
    echo "2) Start services"
    echo "3) Stop services"
    echo "4) Restart services"
    echo "5) Show logs"
    echo "6) Test connections"
    echo "7) Show resource usage"
    echo "8) Manage runlevels"
    echo "9) Exit"
    echo ""
}

# Function to manage runlevels
manage_runlevels() {
    print_header "Runlevel Management"
    
    echo "Current runlevel services:"
    rc-update show default | grep -E "(qdrant|neo4j)" || echo "No services found in default runlevel"
    echo ""
    
    echo "Choose action:"
    echo "1) Add both services to default runlevel"
    echo "2) Remove both services from default runlevel"
    echo "3) Add Qdrant to default runlevel"
    echo "4) Remove Qdrant from default runlevel"
    echo "5) Add Neo4j to default runlevel"
    echo "6) Remove Neo4j from default runlevel"
    echo "7) Show all runlevels"
    read -p "Enter choice (1-7): " choice
    
    case $choice in
        1)
            sudo rc-update add qdrant default
            sudo rc-update add neo4j default
            print_success "Both services added to default runlevel"
            ;;
        2)
            sudo rc-update del qdrant default
            sudo rc-update del neo4j default
            print_success "Both services removed from default runlevel"
            ;;
        3)
            sudo rc-update add qdrant default
            print_success "Qdrant added to default runlevel"
            ;;
        4)
            sudo rc-update del qdrant default
            print_success "Qdrant removed from default runlevel"
            ;;
        5)
            sudo rc-update add neo4j default
            print_success "Neo4j added to default runlevel"
            ;;
        6)
            sudo rc-update del neo4j default
            print_success "Neo4j removed from default runlevel"
            ;;
        7)
            echo "All runlevels:"
            rc-update show
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# Main loop
main() {
    print_header "Qdrant & Neo4j Service Manager"
    
    while true; do
        show_menu
        read -p "Enter choice (1-8): " choice
        
        case $choice in
            1) show_status ;;
            2) start_services ;;
            3) stop_services ;;
            4) restart_services ;;
            5) show_logs ;;
            6) test_connections ;;
            7) show_resources ;;
            8) manage_runlevels ;;
            9) 
                echo "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Check if running as script or sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi