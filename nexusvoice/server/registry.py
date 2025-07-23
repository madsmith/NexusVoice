from typing import Dict, Any, Type, TypeVar, Generic, cast
import asyncio
import logfire

T = TypeVar('T')

class ServiceDescriptor(Generic[T]):
    """
    Descriptor for a registered service
    """
    def __init__(self, service_type: Type[T], provider_name: str, instance: T):
        self.service_type = service_type
        self.provider_name = provider_name
        self.instance = instance
        self.subscribers: set[str] = set()

class ServiceRegistry:
    """
    Central registry for services provided by tasks
    """
    def __init__(self):
        self.services: Dict[str, ServiceDescriptor] = {}
        self._lock = asyncio.Lock()
    
    async def register_service(self, service_id: str, service_type: Type[T], 
                              provider_name: str, instance: T) -> None:
        """
        Register a service with the registry
        """
        async with self._lock:
            if service_id in self.services:
                raise ValueError(f"Service {service_id} already registered")
            
            self.services[service_id] = ServiceDescriptor(service_type, provider_name, instance)
            logfire.info(f"Service registered: {service_id} by {provider_name}")
    
    async def unregister_service(self, service_id: str, provider_name: str) -> None:
        """
        Unregister a service from the registry
        """
        async with self._lock:
            if service_id not in self.services:
                return
            
            descriptor = self.services[service_id]
            if descriptor.provider_name != provider_name:
                raise ValueError(f"Service {service_id} can only be unregistered by its provider")
            
            # Notify subscribers that the service is going away
            for subscriber in list(descriptor.subscribers):
                # In a real implementation, you might want to notify subscribers
                pass
            
            del self.services[service_id]
            logfire.info(f"Service unregistered: {service_id}")
    
    async def get_service(self, service_id: str, expected_type: Type[T], 
                         subscriber_name: str | None = None) -> T | None:
        """
        Get a service from the registry
        """
        async with self._lock:
            if service_id not in self.services:
                return None
            
            descriptor = self.services[service_id]
            if not issubclass(descriptor.service_type, expected_type):
                raise TypeError(f"Service {service_id} is not of expected type {expected_type}")
            
            if subscriber_name:
                descriptor.subscribers.add(subscriber_name)
            
            return cast(T, descriptor.instance)
    
    async def wait_for_service(
        self,
        service_id: str,
        expected_type: Type[T],
        subscriber_name: str,
        timeout: float | None = None
    ) -> T:
        """
        Wait for a service to become available
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            service = await self.get_service(service_id, expected_type, subscriber_name)
            if service:
                return service
            
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Service {service_id} not available after {timeout}s")
            
            await asyncio.sleep(1)
