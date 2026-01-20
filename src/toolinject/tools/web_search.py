"""Web search tool using Tavily."""

from typing import Any

from tavily import TavilyClient

from toolinject.tools.runtime import Tool


class WebSearchTool(Tool):
    """Web search tool with injection capability for adversarial testing."""
    
    name = "web_search"
    description = "Search the web for information. Returns relevant snippets from web pages."
    version = "1.0.0"
    security_notes = "Do not reveal search API configuration or internal parameters."
    
    def __init__(
        self,
        api_key: str,
        inject_content: str | None = None,
        inject_at_position: int = 0,
    ):
        """
        Initialize web search tool.
        
        Args:
            api_key: Tavily API key
            inject_content: Optional content to inject into search results (for adversarial testing)
            inject_at_position: Position to inject content (0 = first result)
        """
        self.api_key = api_key
        self.client = TavilyClient(api_key=api_key) if api_key else None
        self.inject_content = inject_content
        self.inject_at_position = inject_at_position
        self._call_count = 0
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        }
    
    async def execute(self, query: str, max_results: int = 5) -> str:
        """Execute web search."""
        self._call_count += 1
        
        results = []
        
        # If we have injection content, add it
        if self.inject_content:
            if self.inject_at_position == 0:
                results.append({
                    "title": "Relevant Information",
                    "url": "https://example.com/info",
                    "content": self.inject_content,
                })
        
        # If we have a real client, do the search
        if self.client:
            try:
                response = self.client.search(
                    query=query,
                    max_results=max_results,
                )
                
                for i, result in enumerate(response.get("results", [])):
                    # Inject content at specified position
                    if self.inject_content and i + 1 == self.inject_at_position:
                        results.append({
                            "title": "Relevant Information",
                            "url": "https://example.com/info",
                            "content": self.inject_content,
                        })
                    
                    results.append({
                        "title": result.get("title", "Untitled"),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")[:500],
                    })
            except Exception as e:
                return f"Search error: {str(e)}"
        elif not self.inject_content:
            # No client and no injection - return mock results
            results = [
                {
                    "title": f"Result {i+1} for: {query}",
                    "url": f"https://example.com/result{i+1}",
                    "content": f"This is a mock search result for the query: {query}",
                }
                for i in range(min(max_results, 3))
            ]
        
        # Format results
        formatted = []
        for i, r in enumerate(results[:max_results]):
            formatted.append(
                f"[{i+1}] {r['title']}\n"
                f"URL: {r['url']}\n"
                f"{r['content']}\n"
            )
        
        return "\n---\n".join(formatted) if formatted else "No results found."
    
    def set_injection(self, content: str | None, position: int = 0) -> None:
        """Set content to inject into results."""
        self.inject_content = content
        self.inject_at_position = position
    
    def clear_injection(self) -> None:
        """Clear injection content."""
        self.inject_content = None
