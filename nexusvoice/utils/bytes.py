from nexusvoice.utils.logging import get_logger

logger = get_logger(__name__)

class ByteRingBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = bytearray(max_size)
        self.start = 0
        self.end = 0
        self.size = 0

    def append(self, data):
        data_size = len(data)

        if data_size > self.max_size:
            logger.warning(f"Data size {data_size} exceeds buffer size {self.max_size}")
            data = data[-self.max_size:]  # Keep only the last max_size bytes
            data_size = self.max_size

        end = (self.start + self.size) % self.max_size
        data_end = (end + data_size) % self.max_size

        if data_size == self.max_size:
            # Directly overwrite entire buffer
            self.buffer[:] = data
            self.start = 0
            self.end = 0
            self.size = self.max_size
            return

        if end + data_size <= self.max_size:
            # Case 1: Data fits in a single slice
            self.buffer[end:end + data_size] = data
        else:
            # Case 2: Data wraps around
            first_part = self.max_size - end
            self.buffer[end:] = data[:first_part]
            self.buffer[0:data_end] = data[first_part:]

        # Update size and end
        self.size = min(self.size + data_size, self.max_size)
        self.end = data_end
        if self.size == self.max_size:
            self.start = data_end

    def get_bytes(self):
        if self.size == 0:
            return b""
        
        if self.start < self.end:
            return self.buffer[self.start:self.end]
        else:
            return self.buffer[self.start:] + self.buffer[:self.end]

    def byte_count(self):
        return self.size

    def clear(self):
        self.start = 0
        self.end = 0
        self.size = 0

    def __len__(self):
        return self.size

    def __bool__(self):
        return bool(self.size)

    def __repr__(self):
        return f"ByteRingBuffer({self.buffer}, {self.start}, {self.end}, {self.size}/{self.max_size})"

    def __str__(self):
        return f"ByteRingBuffer({self.size}/{self.max_size})"

    def __iter__(self):
        return iter(self.get_bytes())

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.get_bytes()[key]
        else:
            return self.get_bytes()[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("Cannot set individual bytes in ByteRingBuffer")
    

if __name__ == "__main__":
    def show_buffer(buffer):
        bytes = buffer.get_bytes()
        print(f"Buffer: {bytes}")

    buffer = ByteRingBuffer(10)
    print(repr(buffer))
    buffer.append(b"123")
    print(repr(buffer))
    buffer.append(b"456")
    print(repr(buffer))
    buffer.append(b"789")
    print(repr(buffer))
    buffer.append(b"h")
    print(repr(buffer))
    buffer.append(b"e")
    print(repr(buffer))
    buffer.append(b"llo")
    print(repr(buffer))