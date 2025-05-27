#!/usr/bin/env python3
"""
Memory Summarizer for Buddy AI
Handles conversation summarization, episode detection, and smart context expansion
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MemorySummarizer:
    """Handles memory consolidation and summarization"""
    
    def __init__(self, llm, neo4j, qdrant, embedder, collection_name="hermes_advanced_memory"):
        self.llm = llm
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection_name = collection_name
        self.summary_threshold = 100  # interactions before summarization
        self.episode_similarity_threshold = 0.6
        self.log_dir = Path("/var/log/buddy")
        
    def should_summarize(self, interaction_count: int, last_summary_count: int) -> bool:
        """Check if enough interactions have occurred for summarization"""
        return (interaction_count - last_summary_count) >= self.summary_threshold
        
    def detect_episode_boundary(self, current_embedding: np.ndarray, 
                              previous_embedding: Optional[np.ndarray]) -> bool:
        """Detect topic shift using embedding similarity"""
        if previous_embedding is None:
            return False
            
        # Calculate cosine similarity
        similarity = np.dot(current_embedding, previous_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
        )
        
        return similarity < self.episode_similarity_threshold
        
    def load_interaction_range(self, start_line: int, end_line: int) -> List[Dict]:
        """Load interactions from log file by line numbers"""
        interactions = []
        log_file = self.log_dir / "interactions.jsonl"
        
        try:
            with open(log_file, 'r') as f:
                for i, line in enumerate(f):
                    if start_line <= i <= end_line:
                        interactions.append(json.loads(line))
                    elif i > end_line:
                        break
        except Exception as e:
            logger.error(f"Failed to load interaction range: {e}")
            
        return interactions
        
    def generate_summary(self, interactions: List[Dict], summary_type: str = "periodic") -> Dict:
        """Generate a summary of interactions using LLM"""
        # Build context from interactions
        conversation_text = []
        for interaction in interactions:
            conversation_text.append(f"User: {interaction['user_input']}")
            conversation_text.append(f"Assistant: {interaction['assistant_response']}")
            
        conversation = "\n".join(conversation_text)
        
        # Generate summary using LLM
        prompt = f"""Summarize the following conversation, focusing on key facts, decisions, and topics discussed.
Be concise but capture all important information.

Conversation:
{conversation[:2000]}  # Limit context to avoid token limits

Summary:"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.3,
                stop=["\n\n"]
            )
            
            summary_text = response['choices'][0]['text'].strip()
            
            # Extract key topics (simple keyword extraction)
            words = summary_text.lower().split()
            key_topics = [w for w in set(words) if len(w) > 4 and w.isalpha()][:5]
            
            return {
                "text": summary_text,
                "type": summary_type,
                "interaction_count": len(interactions),
                "timestamp_range": {
                    "start": interactions[0]['timestamp'],
                    "end": interactions[-1]['timestamp']
                },
                "log_positions": {
                    "start_line": interactions[0].get('log_line', 0),
                    "end_line": interactions[-1].get('log_line', len(interactions)-1)
                },
                "key_topics": key_topics,
                "episode_ids": list(set(i.get('episode_id', '') for i in interactions))
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
            
    def store_summary_memory(self, summary: Dict, user_id: str = "default_user") -> Optional[str]:
        """Store summary in both Qdrant and Neo4j with metadata"""
        if not summary:
            return None
            
        try:
            import uuid
            from qdrant_client.models import PointStruct
            
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now().timestamp()
            
            # Generate embedding for summary
            embedding = self.embedder.encode([summary['text']])[0]
            
            # Store in Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": summary['text'],
                        "user_id": user_id,
                        "memory_type": "summary",
                        "priority": 0.8,  # Summaries have high priority
                        "timestamp": timestamp,
                        "metadata": {
                            "summary_type": summary['type'],
                            "interaction_count": summary['interaction_count'],
                            "timestamp_range": summary['timestamp_range'],
                            "log_positions": summary['log_positions'],
                            "key_topics": summary['key_topics'],
                            "episode_ids": summary['episode_ids']
                        }
                    }
                )]
            )
            
            # Store in Neo4j
            with self.neo4j.session() as session:
                session.run("""
                    CREATE (m:HermesMemory {
                        id: $memory_id,
                        text: $text,
                        user_id: $user_id,
                        memory_type: 'summary',
                        priority: 0.8,
                        timestamp: $timestamp,
                        summary_type: $summary_type,
                        interaction_count: $interaction_count,
                        start_timestamp: $start_timestamp,
                        end_timestamp: $end_timestamp,
                        log_start_line: $log_start_line,
                        log_end_line: $log_end_line
                    })
                """, 
                memory_id=memory_id,
                text=summary['text'],
                user_id=user_id,
                timestamp=timestamp,
                summary_type=summary['type'],
                interaction_count=summary['interaction_count'],
                start_timestamp=summary['timestamp_range']['start'],
                end_timestamp=summary['timestamp_range']['end'],
                log_start_line=summary['log_positions']['start_line'],
                log_end_line=summary['log_positions']['end_line']
                )
                
            logger.info(f"Stored {summary['type']} summary with {summary['interaction_count']} interactions")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store summary memory: {e}")
            return None
            
    def load_full_context(self, memory_metadata: Dict) -> Optional[str]:
        """Load original interactions from log file for a summary"""
        try:
            log_positions = memory_metadata.get('log_positions', {})
            start_line = log_positions.get('start_line', 0)
            end_line = log_positions.get('end_line', 0)
            
            interactions = self.load_interaction_range(start_line, end_line)
            
            if not interactions:
                return None
                
            # Format interactions for context
            context_lines = []
            for interaction in interactions:
                context_lines.append(f"[{interaction.get('timestamp', 'Unknown time')}]")
                context_lines.append(f"User: {interaction.get('user_input', '')}")
                context_lines.append(f"Assistant: {interaction.get('assistant_response', '')}")
                context_lines.append("")  # Empty line between interactions
                
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"Failed to load full context: {e}")
            return None
            
    def create_episode_summary(self, episode_id: str, user_id: str = "default_user") -> Optional[str]:
        """Create a summary for a completed episode"""
        try:
            # Load all interactions for this episode
            interactions = []
            log_file = self.log_dir / "interactions.jsonl"
            
            with open(log_file, 'r') as f:
                for line in f:
                    interaction = json.loads(line)
                    if interaction.get('episode_id') == episode_id:
                        interactions.append(interaction)
                        
            if len(interactions) < 3:  # Don't summarize very short episodes
                return None
                
            # Generate episode summary
            summary = self.generate_summary(interactions, summary_type="episode")
            
            if summary:
                return self.store_summary_memory(summary, user_id)
                
        except Exception as e:
            logger.error(f"Failed to create episode summary: {e}")
            return None