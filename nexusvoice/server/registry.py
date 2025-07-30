from typing import Any, Type, TypeVar, Generic, cast
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
        self.services: dict[str, ServiceDescriptor] = {}
        self._pending_futures: dict[str, list[asyncio.Future[Any]]] = {}
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
            
            # Complete any pending futures for this service
            if service_id in self._pending_futures:
                for future in self._pending_futures[service_id]:
                    if not future.done():
                        future.set_result(instance)
                del self._pending_futures[service_id]
    
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
            
            # Cancel any pending futures for this service
            if service_id in self._pending_futures:
                for future in self._pending_futures[service_id]:
                    if not future.done():
                        future.cancel()
                del self._pending_futures[service_id]
            
            # Notify subscribers that the service is going away
            for subscriber in list(descriptor.subscribers):
                # In a real implementation, you might want to notify subscribers
                pass
            
            del self.services[service_id]
            logfire.info("Service unregistered: " + service_id)
    
    async def get_service(self, service_id: str, expected_type: Type[T], 
                         subscriber_name: str | None = None) -> T | None:
        """
        Get a service from the registry
        """
        async with self._lock:
            return await self._get_service(service_id, expected_type, subscriber_name)
    
    async def _get_service(self, service_id: str, expected_type: Type[T], 
                         subscriber_name: str | None = None) -> T | None:
        """
        Get a service from the registry
        """
        logfire.trace(f"Getting service {service_id}")
        if service_id not in self.services:
            logfire.debug(f"Service {service_id} not found")
            return None
        
        descriptor = self.services[service_id]
        logfire.trace(f"Service {service_id} found")
        if not issubclass(descriptor.service_type, expected_type):
            logfire.warning(f"Service {service_id} is not of expected type {expected_type}")
            raise TypeError(f"Service {service_id} is not of expected type {expected_type}")
        
        if subscriber_name:
            descriptor.subscribers.add(subscriber_name)
        
        return cast(T, descriptor.instance)
    
    @logfire.instrument("Get Service (Future)")
    async def get_service_future(self, service_id: str, expected_type: Type[T],
                              subscriber_name: str | None = None) -> asyncio.Future[T]:
        """
        Get a future that will be completed when the service becomes available.
        If the service is already available, the future will be completed immediately.
        """
        async with self._lock:
            # Check if the service is already available
            service = await self._get_service(service_id, expected_type, subscriber_name)
            if service:
                # Service is available, return a completed future
                existing_future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
                existing_future.set_result(service)
                return existing_future
            
            # Create a future for the service
            pending_future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
            
            # Add the future to the pending futures for this service
            if service_id not in self._pending_futures:
                self._pending_futures[service_id] = []
            self._pending_futures[service_id].append(pending_future)
            
            return pending_future
    
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
        with logfire.span(f"Waiting for service: {service_id}"):
            # Get a future for the service
            logfire.trace(f"Getting service future for {service_id}")
            future = await self.get_service_future(service_id, expected_type, subscriber_name)
            
            try:
                # Wait for the future to complete with an optional timeout
                logfire.trace(f"Waiting for service {service_id} to become available with timeout {timeout}")
                result = await asyncio.wait_for(future, timeout)
                logfire.trace(f"Service {service_id} became available")
                return result
            except asyncio.TimeoutError:
                # Remove the future from the pending futures
                logfire.warning(f"Service {service_id} not available after {timeout}s")
                async with self._lock:
                    if service_id in self._pending_futures and future in self._pending_futures[service_id]:
                        self._pending_futures[service_id].remove(future)
                        if not self._pending_futures[service_id]:
                            del self._pending_futures[service_id]
                
                raise TimeoutError(f"Service {service_id} not available after {timeout}s")
