from abc import ABC, abstractmethod
import asyncio
import logfire
from typing import TYPE_CHECKING, Dict, Any, Type, TypeVar

if TYPE_CHECKING:
    from nexusvoice.server.NexusServer import NexusServer

from nexusvoice.core.config import NexusConfig
from nexusvoice.server.registry import ServiceRegistry

T = TypeVar('T')

class NexusTask(ABC):
    """Base class for all Nexus tasks."""

    # Class-level declarations for dependencies
    required_services: Dict[str, Type] = {}
    optional_services: Dict[str, Type] = {}
    provided_services: Dict[str, Type] = {}
    
    def __init__(self, server: "NexusServer", config: NexusConfig):
        self.server = server
        self.config = config
        self.running = False

        # Service registries
        self._registry: ServiceRegistry = server.service_registry
        self._required_services: Dict[str, Any] = {}
        self._optional_services: Dict[str, Any] = {}
        self._provided_instances: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """
        Each task will have a name that is used to identify it in the registry.
        """
        return self.__class__.__name__

    @property
    def registry(self) -> ServiceRegistry:
        """
        The service registry for this task.
        """
        # 
        return self._registry
    
    async def initialize(self) -> bool:
        """
        Initialize the task and resolve dependencies.
        """
        # Acquire required services
        for service_id, service_type in self.required_services.items():
            try:
                service = await self.registry.wait_for_service(
                    service_id, service_type, self.name, timeout=10)
                self._required_services[service_id] = service
                logfire.info(f"Task {self.name} acquired required service {service_id}")
            except TimeoutError:
                logfire.error(f"Task {self.name} failed to acquire required service {service_id}")
                return False
                
        # Try to acquire optional services
        for service_id, service_type in self.optional_services.items():
            service = await self.registry.get_service(service_id, service_type, self.name)
            if service:
                self._optional_services[service_id] = service
                logfire.info(f"Task {self.name} acquired optional service {service_id}")
                
        # Register provided services
        self._provided_instances = await self._create_provided_services()
        for service_id, instance in self._provided_instances.items():
            service_type = self.provided_services[service_id]
            await self.registry.register_service(service_id, service_type, self.name, instance)
            
        return True

    async def _create_provided_services(self) -> Dict[str, Any]:
        """Create instances of services provided by this task"""
        # Default implementation returns empty dict
        # Override in subclasses to create actual service instances
        return {}
    
    def register(self):
        """
        Register the task with the server.
        """
        pass

    async def start(self):
        pass

    async def stop(self):
        """Stop the task"""
        self.running = False
        # Unregister provided services
        for service_id in self._provided_instances:
            try:
                await self.registry.unregister_service(service_id, self.name)
            except Exception as e:
                logfire.error(f"Error unregistering service {service_id}: {e}")

    def get_required_service(self, service_id: str, service_type: Type[T] | None = None) -> T:
        """
        Get a required service by ID
        """
        service = self._required_services.get(service_id)
        if not service:
            raise ValueError(f"Required service {service_id} not available")
        if service_type and not isinstance(service, service_type):
            raise TypeError(f"Service {service_id} is not of expected type {service_type}")
        return service
        
    def get_optional_service(self, service_id: str, service_type: Type[T] | None = None) -> T | None:
        """
        Get an optional service by ID
        """
        service = self._optional_services.get(service_id)
        if not service:
            return None
        if service_type and not isinstance(service, service_type):
            raise TypeError(f"Service {service_id} is not of expected type {service_type}")
        return service
