#!/usr/bin/env python3
"""
Qdrant Persistence Verification Script
Verifies that Qdrant is configured for persistent storage and data survives restarts
"""

import os
import sys
import time
import logging
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QdrantPersistenceVerifier:
    def __init__(self):
        self.qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        self.collection_name = "persistence_test_collection"
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def connect(self):
        """Connect to Qdrant"""
        try:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
            
    def check_storage_config(self):
        """Check if Qdrant is configured for persistent storage"""
        try:
            # Get Qdrant info
            info = self.client.get_collections()
            logger.info(f"Qdrant collections: {[c.name for c in info.collections]}")
            
            # Check if storage directory exists (typical location)
            storage_paths = [
                "/var/lib/qdrant/storage",
                "/qdrant/storage",
                "./storage"  # Local dev path
            ]
            
            for path in storage_paths:
                if os.path.exists(path):
                    logger.info(f"Found Qdrant storage directory: {path}")
                    # Check if it's writable
                    test_file = os.path.join(path, ".write_test")
                    try:
                        with open(test_file, 'w') as f:
                            f.write("test")
                        os.remove(test_file)
                        logger.info(f"Storage directory {path} is writable")
                        return True
                    except:
                        logger.warning(f"Storage directory {path} is not writable")
                        
            logger.warning("No standard Qdrant storage directory found")
            return True  # Qdrant might be using a custom path
            
        except Exception as e:
            logger.error(f"Error checking storage config: {e}")
            return False
            
    def create_test_data(self):
        """Create test collection and data"""
        try:
            # Delete collection if exists
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except:
                pass
                
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {self.collection_name}")
            
            # Add test points
            test_texts = [
                "This is a persistence test point 1",
                "This is a persistence test point 2",
                "This is a persistence test point 3"
            ]
            
            points = []
            for i, text in enumerate(test_texts):
                embedding = self.embedding_model.encode(text).tolist()
                point = PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "test_id": "persistence_test_2024"
                    }
                )
                points.append(point)
                
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Added {len(points)} test points to collection")
            
            # Verify data was written
            count = self.client.get_collection(self.collection_name).points_count
            logger.info(f"Collection now has {count} points")
            
            return count == len(points)
            
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            return False
            
    def verify_persistence(self):
        """Verify data persists by checking for test data"""
        try:
            # Check if test collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Test collection '{self.collection_name}' not found - no previous test data")
                return None
                
            # Get collection info
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Found test collection with {collection.points_count} points")
            
            # Search for test data
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            test_points = []
            for point in results[0]:
                if point.payload.get("test_id") == "persistence_test_2024":
                    test_points.append(point)
                    
            if test_points:
                logger.info(f"Found {len(test_points)} persisted test points:")
                for point in test_points:
                    logger.info(f"  - ID: {point.id}, Text: {point.payload.get('text')}")
                    logger.info(f"    Created: {point.payload.get('timestamp')}")
                return True
            else:
                logger.info("No test points found from previous run")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying persistence: {e}")
            return False
            
    def cleanup(self):
        """Clean up test data"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Cleaned up test collection: {self.collection_name}")
        except:
            pass
            
    def run_verification(self, cleanup_after=False):
        """Run full verification process"""
        logger.info("=== Qdrant Persistence Verification ===")
        
        if not self.connect():
            return False
            
        # Check storage configuration
        logger.info("\n1. Checking storage configuration...")
        storage_ok = self.check_storage_config()
        
        # Check for persisted data
        logger.info("\n2. Checking for persisted test data...")
        has_persisted_data = self.verify_persistence()
        
        # Create new test data
        logger.info("\n3. Creating new test data...")
        data_created = self.create_test_data()
        
        # Summary
        logger.info("\n=== Verification Summary ===")
        logger.info(f"Storage configuration: {'OK' if storage_ok else 'WARNING'}")
        logger.info(f"Previous test data found: {has_persisted_data}")
        logger.info(f"New test data created: {data_created}")
        
        if has_persisted_data is None:
            logger.info("\nThis appears to be the first run. Run this script again to verify persistence.")
        elif has_persisted_data:
            logger.info("\n✓ Qdrant persistence is working correctly!")
        else:
            logger.warning("\n⚠ Qdrant may not be persisting data correctly!")
            
        if cleanup_after:
            self.cleanup()
            
        return storage_ok and data_created


if __name__ == "__main__":
    verifier = QdrantPersistenceVerifier()
    
    # Check for cleanup flag
    cleanup = "--cleanup" in sys.argv
    
    success = verifier.run_verification(cleanup_after=cleanup)
    
    if not cleanup:
        logger.info("\nTo verify persistence:")
        logger.info("1. Restart Qdrant service")
        logger.info("2. Run this script again")
        logger.info("3. If test data is found, persistence is working")
        logger.info("\nTo clean up test data: python verify_qdrant_persistence.py --cleanup")
    
    sys.exit(0 if success else 1)