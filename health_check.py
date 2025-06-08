"""
System Health Checker for Buddy Memory System
Validates system health and component status
NO silent failures - all errors are loud and proud
"""

from typing import Dict, List, Any
import requests
import time
from pathlib import Path

# Rule 3 Compliance - Exception hierarchy
from exceptions import HealthCheckError
from debug_info import debug_tracker


class SystemHealthChecker:
    """Validates system health and component status"""
    
    def __init__(self):
        """Initialize health checker with component configurations"""
        self.neo4j_url = "http://localhost:7474"
        self.qdrant_url = "http://localhost:6333"
        self.neo4j_auth = ("neo4j", "password123")
        
        # Log directory for file system checks
        self.log_dir = Path("/var/log/buddy/full_context")
    
    def check_neo4j(self) -> Dict[str, Any]:
        """
        Check Neo4j connectivity and health
        Returns health status dictionary
        """
        try:
            start_time = time.time()
            
            # Check basic connectivity
            response = requests.get(f"{self.neo4j_url}/", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            # Try a simple query to verify database functionality
            query_start = time.time()
            query_response = requests.post(
                f"{self.neo4j_url}/db/neo4j/tx/commit",
                auth=self.neo4j_auth,
                headers={"Content-Type": "application/json"},
                json={"statements": [{"statement": "RETURN 1 as test"}]},
                timeout=5
            )
            query_time = (time.time() - query_start) * 1000
            
            # Determine health status
            is_healthy = (response.status_code == 200 and 
                         query_response.status_code == 200)
            
            result = {
                "status": "healthy" if is_healthy else "unhealthy",
                "response_time_ms": response_time,
                "query_time_ms": query_time,
                "http_status": response.status_code,
                "query_status": query_response.status_code,
                "url": self.neo4j_url,
                "error": None
            }
            
            # Add query result details if successful
            if query_response.status_code == 200:
                query_data = query_response.json()
                result["query_successful"] = True
                result["query_result"] = query_data
            else:
                result["query_successful"] = False
                result["query_error"] = f"Query failed with status {query_response.status_code}"
            
            return result
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time_ms": None,
                "query_time_ms": None,
                "http_status": None,
                "query_status": None,
                "query_successful": False,
                "url": self.neo4j_url,
                "error": str(e)
            }
    
    def check_qdrant(self) -> Dict[str, Any]:
        """
        Check Qdrant connectivity and health
        Returns health status dictionary
        """
        try:
            start_time = time.time()
            
            # Check collections endpoint
            response = requests.get(f"{self.qdrant_url}/collections", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            is_healthy = response.status_code == 200
            
            result = {
                "status": "healthy" if is_healthy else "unhealthy",
                "response_time_ms": response_time,
                "http_status": response.status_code,
                "url": self.qdrant_url,
                "error": None
            }
            
            # Add collection details if successful
            if response.status_code == 200:
                try:
                    collections_data = response.json()
                    result["collections"] = collections_data
                    result["collections_count"] = len(collections_data.get("result", {}).get("collections", []))
                except Exception as parse_error:
                    result["collections"] = None
                    result["collections_count"] = 0
                    result["parse_error"] = str(parse_error)
            else:
                result["collections"] = None
                result["collections_count"] = 0
                
            return result
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "response_time_ms": None,
                "http_status": None,
                "collections": None,
                "collections_count": 0,
                "url": self.qdrant_url,
                "error": str(e)
            }
    
    def check_file_system(self) -> Dict[str, Any]:
        """
        Check file system access for logging
        Returns health status dictionary
        """
        try:
            # Check if directory exists
            directory_exists = self.log_dir.exists()
            
            if not directory_exists:
                return {
                    "status": "unhealthy",
                    "directory_exists": False,
                    "directory_path": str(self.log_dir),
                    "writable": False,
                    "readable": False,
                    "error": f"Log directory does not exist: {self.log_dir}"
                }
            
            # Check if directory is readable
            try:
                list(self.log_dir.iterdir())
                readable = True
            except Exception:
                readable = False
            
            # Check if directory is writable by trying to create and delete a test file
            writable = False
            write_error = None
            try:
                test_file = self.log_dir / "health_check_test.txt"
                test_content = f"Health check test at {time.time()}"
                test_file.write_text(test_content)
                
                # Verify we can read it back
                read_content = test_file.read_text()
                if read_content == test_content:
                    writable = True
                
                # Clean up test file
                test_file.unlink()
                
            except Exception as e:
                write_error = str(e)
            
            is_healthy = directory_exists and readable and writable
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "directory_exists": directory_exists,
                "directory_path": str(self.log_dir),
                "readable": readable,
                "writable": writable,
                "error": write_error if write_error else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "directory_exists": False,
                "directory_path": str(self.log_dir),
                "readable": False,
                "writable": False,
                "error": str(e)
            }
    
    def check_debug_system(self) -> Dict[str, Any]:
        """
        Check debug system functionality
        Returns health status dictionary
        """
        try:
            start_time = time.time()
            
            # Test debug tracker functionality
            test_operation_id = f"health_check_{int(time.time())}"
            
            # Test operation logging
            debug_tracker.log_operation("health_checker", "debug_system_test", {
                "test_id": test_operation_id
            })
            
            # Test statistics generation
            stats = debug_tracker.get_statistics()
            
            # Test recent operations retrieval
            recent_ops = debug_tracker.get_recent_operations(5)
            
            # Test debug summary generation
            summary = debug_tracker.get_debug_summary()
            
            check_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "check_time_ms": check_time,
                "total_operations": stats["total_operations"],
                "components_tracked": stats["components_tracked"],
                "recent_operations_count": len(recent_ops),
                "summary_generated": "total_operations" in summary,
                "error": None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "check_time_ms": None,
                "total_operations": None,
                "components_tracked": None,
                "recent_operations_count": None,
                "summary_generated": False,
                "error": str(e)
            }
    
    def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check of all components
        Returns complete system health status
        FAILS LOUDLY if health check system itself fails
        """
        try:
            debug_tracker.log_operation("health_checker", "comprehensive_check_start", {})
            
            start_time = time.time()
            
            # Run individual component checks
            results = {
                "timestamp": time.time(),
                "check_duration_ms": 0,
                "overall_status": "healthy",
                "components": {
                    "neo4j": self.check_neo4j(),
                    "qdrant": self.check_qdrant(),
                    "file_system": self.check_file_system(),
                    "debug_system": self.check_debug_system()
                }
            }
            
            # Calculate check duration
            results["check_duration_ms"] = (time.time() - start_time) * 1000
            
            # Determine overall status
            unhealthy_components = [name for name, status in results["components"].items() 
                                  if status["status"] != "healthy"]
            
            if unhealthy_components:
                results["overall_status"] = "unhealthy"
                results["unhealthy_components"] = unhealthy_components
                results["unhealthy_count"] = len(unhealthy_components)
            else:
                results["unhealthy_components"] = []
                results["unhealthy_count"] = 0
            
            # Add summary statistics
            results["healthy_count"] = len(results["components"]) - len(unhealthy_components)
            results["total_components"] = len(results["components"])
            
            # Log completion
            debug_tracker.log_operation("health_checker", "comprehensive_check_complete", {
                "overall_status": results["overall_status"],
                "unhealthy_components": unhealthy_components,
                "check_duration_ms": results["check_duration_ms"],
                "total_components": results["total_components"]
            }, success=(results["overall_status"] == "healthy"))
            
            return results
            
        except Exception as e:
            # Log failure
            debug_tracker.log_operation("health_checker", "comprehensive_check_failed", {
                "error_type": type(e).__name__
            }, success=False, error=str(e))
            
            # LOUD FAILURE per Rule 3
            raise HealthCheckError(f"Comprehensive health check failed: {e}") from e
    
    def get_health_summary(self) -> str:
        """
        Get a human-readable health summary
        FAILS LOUDLY if summary generation fails
        """
        try:
            health_results = self.comprehensive_health_check()
            
            summary_lines = []
            summary_lines.append(f"=== SYSTEM HEALTH SUMMARY ===")
            summary_lines.append(f"Overall Status: {health_results['overall_status'].upper()}")
            summary_lines.append(f"Check Duration: {health_results['check_duration_ms']:.1f}ms")
            summary_lines.append(f"Components: {health_results['healthy_count']}/{health_results['total_components']} healthy")
            summary_lines.append("")
            
            for component, status in health_results['components'].items():
                status_icon = "✓" if status['status'] == 'healthy' else "✗"
                summary_lines.append(f"{status_icon} {component}: {status['status']}")
                
                # Add component-specific details
                if status.get('response_time_ms'):
                    summary_lines.append(f"   Response time: {status['response_time_ms']:.1f}ms")
                
                if status.get('collections_count') is not None:
                    summary_lines.append(f"   Collections: {status['collections_count']}")
                
                if status.get('total_operations') is not None:
                    summary_lines.append(f"   Operations tracked: {status['total_operations']}")
                
                if status.get('error'):
                    summary_lines.append(f"   Error: {status['error']}")
                
                summary_lines.append("")
            
            return "\\n".join(summary_lines)
            
        except Exception as e:
            raise HealthCheckError(f"Health summary generation failed: {e}") from e


# Global health checker instance
health_checker = SystemHealthChecker()