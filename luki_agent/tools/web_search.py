"""
Web search tool using Tavily API for real-time information retrieval.

This tool enables LUKi to search the web for current information, recent news,
and facts that require up-to-date knowledge beyond the model's training data.
"""

import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Tavily-based web search tool optimized for LLM consumption.
    
    This tool provides:
    - Real-time web search for current information
    - LLM-optimized result formatting
    - AI-generated answer summaries
    - Source attribution for verification
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.
        
        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
            
        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key required. Set TAVILY_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("✅ WebSearchTool initialized successfully")
        except ImportError:
            raise ImportError(
                "tavily-python not installed. Run: pip install tavily-python==0.3.0"
            )
    
    def search(
        self, 
        query: str, 
        max_results: int = 3,
        search_depth: str = "basic"
    ) -> Dict[str, Any]:
        """
        Search the web for current information.
        
        Args:
            query: Search query (e.g., "UK Prime Minister 2025", "latest iPhone features")
            max_results: Number of results to return (1-5). Default is 3.
            search_depth: "basic" for faster results, "advanced" for deeper search
            
        Returns:
            Dictionary containing:
                - success (bool): Whether search succeeded
                - query (str): Original search query
                - answer (str): AI-generated summary from Tavily
                - results (List[Dict]): Search results with title, url, content, score
                - sources (List[str]): List of source URLs
                - error (str, optional): Error message if search failed
        """
        try:
            logger.info(f"🔍 Executing web search: {query}")
            
            # Call Tavily API
            results = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,  # Tavily's AI-generated summary
                include_raw_content=False,  # We want cleaned snippets
                include_images=False  # Focus on text for now
            )
            
            # Format results for LLM consumption
            formatted_results = {
                "success": True,
                "query": query,
                "answer": results.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:600],  # Limit snippet length
                        "score": r.get("score", 0.0)
                    }
                    for r in results.get("results", [])
                ],
                "sources": [r.get("url", "") for r in results.get("results", [])],
                "search_depth": search_depth
            }
            
            logger.info(f"✅ Search completed: {len(formatted_results['results'])} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "answer": "",
                "results": [],
                "sources": []
            }
    
    @staticmethod
    def get_tool_definition() -> Dict[str, Any]:
        """
        Return the tool definition for function calling.
        
        This definition tells the LLM when and how to use the web search tool.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for current information, recent news, facts, or events. "
                    "Use this ONLY when the user's question requires up-to-date information "
                    "that is likely beyond your training data cutoff. "
                    "\n\nUse for: current events, recent news, latest products, current leaders/politicians, "
                    "real-time data, recent scientific discoveries. "
                    "\n\nDO NOT use for: historical facts, general knowledge, platform information "
                    "(use your core knowledge instead), or user's personal information (use ELR memories)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search query. Be specific and include context. "
                                "Examples: 'UK Prime Minister 2025', 'latest iPhone model features', "
                                "'recent AI breakthroughs 2025', 'Super Bowl winner 2025'"
                            )
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of search results to return (1-5). Default is 3.",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def format_results_for_llm(self, search_results: Dict[str, Any]) -> str:
        """
        Format search results into a clean string for LLM context.
        
        Args:
            search_results: Results from self.search()
            
        Returns:
            Formatted string ready for LLM consumption
        """
        if not search_results.get("success"):
            return f"Search failed: {search_results.get('error', 'Unknown error')}"
        
        formatted = f"Web Search Results for: {search_results['query']}\n\n"
        
        # Add Tavily's AI summary if available
        if search_results.get("answer"):
            formatted += f"Summary: {search_results['answer']}\n\n"
        
        # Add individual results
        formatted += "Detailed Results:\n"
        for i, result in enumerate(search_results.get("results", []), 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   Source: {result['url']}\n"
            formatted += f"   {result['content']}\n\n"
        
        # Add sources footer
        sources = search_results.get("sources", [])
        if sources:
            formatted += f"Sources: {', '.join(sources[:3])}"
        
        return formatted
