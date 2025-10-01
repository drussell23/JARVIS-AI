#!/usr/bin/env python3
"""
Zero Configuration Mesh Network for JARVIS
Enables automatic service discovery and interconnection
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
import socket
import json

logger = logging.getLogger(__name__)


class ZeroConfigMesh:
    """
    Zero-configuration mesh network for service discovery
    Services automatically find and connect to each other
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._discovery_task = None

    async def start(self):
        """Start the mesh network"""
        self._running = True
        logger.info("✅ Zero-config mesh network started")

        # Start discovery background task
        self._discovery_task = asyncio.create_task(self._periodic_discovery())

    async def stop(self):
        """Stop the mesh network"""
        self._running = False

        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        logger.info("Zero-config mesh network stopped")

    async def join(self, service_info: Dict[str, Any]):
        """
        Add a service to the mesh network

        Args:
            service_info: Service information including name, port, protocol
        """
        name = service_info.get("name")
        if not name:
            logger.error("Cannot join mesh: service info missing 'name'")
            return

        service_info["joined_at"] = datetime.now().isoformat()
        service_info["last_seen"] = datetime.now().isoformat()

        self.nodes[name] = service_info
        logger.info(f"✅ Service '{name}' joined mesh network")

    async def leave(self, service_name: str):
        """Remove a service from the mesh network"""
        if service_name in self.nodes:
            del self.nodes[service_name]
            logger.info(f"Service '{service_name}' left mesh network")

    async def find_service(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a service in the mesh network by name

        Args:
            name: Service name to find

        Returns:
            Service info if found, None otherwise
        """
        return self.nodes.get(name)

    async def find_services_by_type(self, service_type: str) -> List[Dict[str, Any]]:
        """
        Find all services of a given type

        Args:
            service_type: Type of service to find (e.g., "backend", "vision", "voice")

        Returns:
            List of matching services
        """
        matching = []
        for service in self.nodes.values():
            if service.get("type") == service_type:
                matching.append(service)
        return matching

    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all services in the mesh"""
        return self.nodes.copy()

    async def _periodic_discovery(self):
        """Periodically update service discovery"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every 60 seconds

                # Update last_seen for active nodes
                now = datetime.now().isoformat()
                for name, service in self.nodes.items():
                    # Check if service is still reachable
                    if await self._check_service_reachable(service):
                        service["last_seen"] = now
                    else:
                        logger.warning(f"Service '{name}' is unreachable")

                # Remove stale nodes (not seen for 5 minutes)
                stale_nodes = []
                for name, service in self.nodes.items():
                    last_seen = datetime.fromisoformat(service.get("last_seen", now))
                    age_seconds = (datetime.now() - last_seen).total_seconds()
                    if age_seconds > 300:  # 5 minutes
                        stale_nodes.append(name)

                for name in stale_nodes:
                    logger.warning(f"Removing stale node: {name}")
                    await self.leave(name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic discovery: {e}")

    async def _check_service_reachable(self, service: Dict[str, Any]) -> bool:
        """Check if a service is reachable"""
        try:
            port = service.get("port")
            if not port:
                return False

            # Try to connect to the port
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", port),
                timeout=2
            )
            writer.close()
            await writer.wait_closed()
            return True

        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get mesh network status"""
        return {
            "running": self._running,
            "node_count": len(self.nodes),
            "nodes": {
                name: {
                    "port": service.get("port"),
                    "protocol": service.get("protocol"),
                    "type": service.get("type"),
                    "joined_at": service.get("joined_at"),
                    "last_seen": service.get("last_seen")
                }
                for name, service in self.nodes.items()
            }
        }

    async def get_mesh_config(self) -> Dict[str, Any]:
        """Get mesh configuration and topology"""
        return {
            "enabled": self._running,
            "node_count": len(self.nodes),
            "topology": "peer-to-peer",
            "discovery_interval_seconds": 60,
            "stale_timeout_seconds": 300,
            "nodes": self.nodes,
            "stats": {
                "total_nodes": len(self.nodes),
                "active_nodes": sum(1 for node in self.nodes.values() if node.get("last_seen")),
                "service_types": len(set(node.get("type") for node in self.nodes.values() if node.get("type"))),
                "total_connections": len(self.nodes) * (len(self.nodes) - 1) if len(self.nodes) > 1 else 0  # Mesh topology
            }
        }

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all services in the mesh

        Args:
            message: Message to broadcast
        """
        logger.info(f"Broadcasting message to {len(self.nodes)} nodes")
        for name, service in self.nodes.items():
            try:
                # In a full implementation, this would send the message
                # For now, just log it
                logger.debug(f"Would broadcast to {name}: {message}")
            except Exception as e:
                logger.error(f"Failed to broadcast to {name}: {e}")


# Global mesh instance
_mesh: Optional[ZeroConfigMesh] = None


def get_mesh() -> ZeroConfigMesh:
    """Get or create the global mesh instance"""
    global _mesh
    if _mesh is None:
        _mesh = ZeroConfigMesh()
    return _mesh
