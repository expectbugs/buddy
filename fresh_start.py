#!/usr/bin/env python3
"""
Fresh Start Script for Buddy Memory System
Clears all memory data and context logs for a completely clean start

This script:
1. Clears Neo4j database (all nodes and relationships)
2. Recreates Qdrant collection (deletes and recreates)  
3. Deletes all context logs and lookup index
4. Provides verification of clean state

USAGE:
  python fresh_start.py                    # Interactive mode with confirmation
  python fresh_start.py --confirm          # Automatic mode (no confirmation)
  python fresh_start.py --check-only       # Check current state only
  python fresh_start.py --help             # Show help

EXAMPLES:
  # Check current state
  python fresh_start.py --check-only
  
  # Clean everything with confirmation
  python fresh_start.py
  
  # Clean everything automatically (for scripts)
  python fresh_start.py --confirm

REQUIREMENTS:
- Neo4j running on localhost:7474 (username: neo4j, password: password123)
- Qdrant running on localhost:6333
- Write access to /var/log/buddy/full_context/

Rules Compliance:
- Rule 3: NO silent failures - all operations fail loudly
- Rule 4: Uses officially recommended database APIs
- Rule 5: Clean operation with verification
"""

import os
import sys
import logging
import requests
import json
import shutil
from pathlib import Path
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshStartCleaner:
    """
    Handles complete cleanup of the Buddy memory system
    Clears databases and logs for a fresh start
    """
    
    def __init__(self):
        # Database connection settings
        self.neo4j_url = "http://localhost:7474/db/neo4j/tx/commit"
        self.neo4j_auth = ("neo4j", "password123")
        self.qdrant_url = "http://localhost:6333"
        self.collection_name = "mem0_fixed"
        self.context_log_dir = Path("/var/log/buddy/full_context")
        
        logger.info("FreshStartCleaner initialized")
    
    def check_services(self) -> bool:
        """
        Check if Neo4j and Qdrant services are running
        FAILS LOUDLY if services are not accessible
        """
        logger.info("Checking database services...")
        
        # Check Neo4j
        try:
            response = requests.post(
                self.neo4j_url,
                auth=self.neo4j_auth,
                headers={"Content-Type": "application/json"},
                json={"statements": [{"statement": "RETURN 1"}]},
                timeout=5
            )
            response.raise_for_status()
            logger.info("✓ Neo4j service is accessible")
        except Exception as e:
            raise RuntimeError(f"NEO4J SERVICE NOT ACCESSIBLE: {e}")
        
        # Check Qdrant
        try:
            response = requests.get(f"{self.qdrant_url}/collections", timeout=5)
            response.raise_for_status()
            logger.info("✓ Qdrant service is accessible")
        except Exception as e:
            raise RuntimeError(f"QDRANT SERVICE NOT ACCESSIBLE: {e}")
        
        return True
    
    def clear_neo4j(self) -> None:
        """
        Clear all data from Neo4j database
        FAILS LOUDLY if clearing fails
        """
        logger.info("Clearing Neo4j database...")
        
        # First clear relationships
        try:
            response = requests.post(
                self.neo4j_url,
                auth=self.neo4j_auth,
                headers={"Content-Type": "application/json"},
                json={"statements": [{"statement": "MATCH ()-[r]-() DELETE r"}]},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("errors"):
                raise RuntimeError(f"NEO4J RELATIONSHIP DELETION FAILED: {result['errors']}")
            
            logger.info("✓ Neo4j relationships cleared")
        except Exception as e:
            raise RuntimeError(f"FAILED TO CLEAR NEO4J RELATIONSHIPS: {e}")
        
        # Then clear nodes
        try:
            response = requests.post(
                self.neo4j_url,
                auth=self.neo4j_auth,
                headers={"Content-Type": "application/json"},
                json={"statements": [{"statement": "MATCH (n) DELETE n"}]},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("errors"):
                raise RuntimeError(f"NEO4J NODE DELETION FAILED: {result['errors']}")
            
            logger.info("✓ Neo4j nodes cleared")
        except Exception as e:
            raise RuntimeError(f"FAILED TO CLEAR NEO4J NODES: {e}")
        
        # Verify clearing
        try:
            response = requests.post(
                self.neo4j_url,
                auth=self.neo4j_auth,
                headers={"Content-Type": "application/json"},
                json={"statements": [{"statement": "MATCH (n) RETURN count(n) as node_count"}]},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            node_count = result["results"][0]["data"][0]["row"][0]
            if node_count != 0:
                raise RuntimeError(f"NEO4J CLEARING VERIFICATION FAILED: {node_count} nodes remain")
            
            logger.info(f"✓ Neo4j database verified clean: {node_count} nodes")
        except Exception as e:
            raise RuntimeError(f"FAILED TO VERIFY NEO4J CLEARING: {e}")
    
    def clear_qdrant(self) -> None:
        """
        Clear Qdrant collection by deleting and recreating
        FAILS LOUDLY if operation fails
        """
        logger.info(f"Clearing Qdrant collection: {self.collection_name}")
        
        # Delete collection
        try:
            response = requests.delete(f"{self.qdrant_url}/collections/{self.collection_name}", timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if not result.get("result"):
                raise RuntimeError(f"QDRANT COLLECTION DELETION FAILED: {result}")
            
            logger.info(f"✓ Qdrant collection '{self.collection_name}' deleted")
        except Exception as e:
            raise RuntimeError(f"FAILED TO DELETE QDRANT COLLECTION: {e}")
        
        # Recreate collection
        try:
            response = requests.put(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                headers={"Content-Type": "application/json"},
                json={"vectors": {"size": 1536, "distance": "Cosine"}},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if not result.get("result"):
                raise RuntimeError(f"QDRANT COLLECTION CREATION FAILED: {result}")
            
            logger.info(f"✓ Qdrant collection '{self.collection_name}' recreated")
        except Exception as e:
            raise RuntimeError(f"FAILED TO RECREATE QDRANT COLLECTION: {e}")
        
        # Verify collection is empty
        try:
            response = requests.get(f"{self.qdrant_url}/collections/{self.collection_name}", timeout=10)
            response.raise_for_status()
            result = response.json()
            
            points_count = result["result"]["points_count"]
            if points_count != 0:
                raise RuntimeError(f"QDRANT CLEARING VERIFICATION FAILED: {points_count} points remain")
            
            logger.info(f"✓ Qdrant collection verified clean: {points_count} points")
        except Exception as e:
            raise RuntimeError(f"FAILED TO VERIFY QDRANT CLEARING: {e}")
    
    def clear_context_logs(self) -> None:
        """
        Clear all context logs and lookup index
        FAILS LOUDLY if clearing fails
        """
        logger.info(f"Clearing context logs from: {self.context_log_dir}")
        
        if not self.context_log_dir.exists():
            logger.info("✓ Context log directory does not exist (already clean)")
            return
        
        try:
            # Count files before deletion
            files_before = list(self.context_log_dir.glob("*"))
            file_count = len(files_before)
            
            if file_count == 0:
                logger.info("✓ Context log directory already clean")
                return
            
            # Delete all files in the directory
            for file_path in files_before:
                if file_path.is_file():
                    file_path.unlink()
                    logger.debug(f"Deleted: {file_path.name}")
            
            # Verify deletion
            files_after = list(self.context_log_dir.glob("*"))
            if files_after:
                raise RuntimeError(f"CONTEXT LOG CLEARING FAILED: {len(files_after)} files remain")
            
            logger.info(f"✓ Context logs cleared: {file_count} files deleted")
            
        except Exception as e:
            raise RuntimeError(f"FAILED TO CLEAR CONTEXT LOGS: {e}")
    
    def verify_clean_state(self) -> Dict[str, int]:
        """
        Verify that all systems are in clean state
        Returns counts for verification
        FAILS LOUDLY if verification fails
        """
        logger.info("Verifying clean state...")
        
        verification = {}
        
        # Verify Neo4j
        try:
            response = requests.post(
                self.neo4j_url,
                auth=self.neo4j_auth,
                headers={"Content-Type": "application/json"},
                json={"statements": [{"statement": "MATCH (n) RETURN count(n) as nodes"}]},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            verification["neo4j_nodes"] = result["results"][0]["data"][0]["row"][0]
        except Exception as e:
            raise RuntimeError(f"NEO4J VERIFICATION FAILED: {e}")
        
        # Verify Qdrant
        try:
            response = requests.get(f"{self.qdrant_url}/collections/{self.collection_name}", timeout=10)
            response.raise_for_status()
            result = response.json()
            verification["qdrant_points"] = result["result"]["points_count"]
        except Exception as e:
            raise RuntimeError(f"QDRANT VERIFICATION FAILED: {e}")
        
        # Verify context logs
        try:
            if self.context_log_dir.exists():
                verification["context_log_files"] = len(list(self.context_log_dir.glob("*")))
            else:
                verification["context_log_files"] = 0
        except Exception as e:
            raise RuntimeError(f"CONTEXT LOG VERIFICATION FAILED: {e}")
        
        # Check if everything is clean
        total_items = sum(verification.values())
        if total_items != 0:
            raise RuntimeError(f"CLEAN STATE VERIFICATION FAILED: {verification}")
        
        logger.info("✅ Clean state verified - all systems clean")
        return verification
    
    def run_fresh_start(self, confirm: bool = False) -> None:
        """
        Run complete fresh start process
        
        Args:
            confirm: If True, skip confirmation prompt
        """
        logger.info("🧹 Starting Fresh Start Process...")
        logger.info("This will permanently delete ALL memory data and context logs!")
        
        if not confirm:
            response = input("\nAre you sure you want to proceed? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Fresh start cancelled by user")
                return
        
        try:
            # Step 1: Check services
            self.check_services()
            
            # Step 2: Clear databases
            self.clear_neo4j()
            self.clear_qdrant()
            
            # Step 3: Clear logs
            self.clear_context_logs()
            
            # Step 4: Verify clean state
            verification = self.verify_clean_state()
            
            # Success!
            logger.info("🎉 Fresh start completed successfully!")
            logger.info("System Status:")
            logger.info(f"  Neo4j nodes: {verification['neo4j_nodes']}")
            logger.info(f"  Qdrant points: {verification['qdrant_points']}")
            logger.info(f"  Context log files: {verification['context_log_files']}")
            logger.info("✅ Ready for fresh start with Phase 2 Enhanced Memory!")
            
        except Exception as e:
            logger.error(f"❌ Fresh start failed: {e}")
            raise

def main():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fresh start script for Buddy memory system")
    parser.add_argument(
        "--confirm", 
        action="store_true", 
        help="Skip confirmation prompt and proceed automatically"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check services and current state, don't clean anything"
    )
    
    args = parser.parse_args()
    
    cleaner = FreshStartCleaner()
    
    try:
        if args.check_only:
            logger.info("🔍 Checking system state...")
            cleaner.check_services()
            verification = cleaner.verify_clean_state()
            logger.info("Current state:")
            logger.info(f"  Neo4j nodes: {verification['neo4j_nodes']}")
            logger.info(f"  Qdrant points: {verification['qdrant_points']}")
            logger.info(f"  Context log files: {verification['context_log_files']}")
        else:
            cleaner.run_fresh_start(confirm=args.confirm)
    except KeyboardInterrupt:
        logger.info("Fresh start interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fresh start failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()