"""Health checks for fleet providers — probes real endpoints."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin

import urllib.request
import urllib.error


@dataclass
class HealthStatus:
    """Health check result for a provider."""
    provider: str
    url: str
    healthy: bool
    latency_ms: float = 0.0
    error: str = ""
    model: str = ""


class FleetHealthChecker:
    """Probes fleet providers to check availability."""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    def check_provider(self, name: str, host: str, port: int, path: str = "/v1/models") -> HealthStatus:
        """Check a single provider."""
        url = f"http://{host}:{port}{path}"
        start = time.time()
        
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "SentinelCLI/2.0")
            
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                latency = (time.time() - start) * 1000
                data = resp.read().decode("utf-8", errors="replace")
                
                # Try to extract model name
                model = ""
                try:
                    import json
                    parsed = json.loads(data)
                    if isinstance(parsed, dict) and "data" in parsed:
                        models = parsed["data"]
                        if isinstance(models, list) and models:
                            model = models[0].get("id", models[0].get("name", ""))
                except Exception:
                    pass
                
                return HealthStatus(
                    provider=name,
                    url=url,
                    healthy=True,
                    latency_ms=latency,
                    model=model,
                )
        except urllib.error.URLError as e:
            return HealthStatus(
                provider=name,
                url=url,
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                error=str(e.reason if hasattr(e, "reason") else e),
            )
        except Exception as e:
            return HealthStatus(
                provider=name,
                url=url,
                healthy=False,
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    def check_all(self) -> dict[str, HealthStatus]:
        """Check all fleet providers."""
        from .fleet_router import FLEET_PROVIDERS
        
        results = {}
        for name, provider in FLEET_PROVIDERS.items():
            if provider.is_local:
                status = self.check_provider(name, provider.host, provider.port)
                results[name] = status
        
        return results
    
    def get_best_available(self, prefer_local: bool = True) -> Optional[HealthStatus]:
        """Get the best available provider."""
        results = self.check_all()
        
        # Filter healthy ones
        healthy = [s for s in results.values() if s.healthy]
        if not healthy:
            return None
        
        # Sort by latency
        healthy.sort(key=lambda s: s.latency_ms)
        return healthy[0]
    
    def summary(self) -> dict[str, Any]:
        """Get a summary of all provider health."""
        results = self.check_all()
        return {
            "total": len(results),
            "healthy": sum(1 for s in results.values() if s.healthy),
            "unhealthy": sum(1 for s in results.values() if not s.healthy),
            "providers": {
                name: {
                    "healthy": s.healthy,
                    "latency_ms": round(s.latency_ms, 1),
                    "model": s.model,
                    "error": s.error,
                }
                for name, s in results.items()
            },
        }
