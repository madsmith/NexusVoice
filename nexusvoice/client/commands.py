class Command:
    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return self.__str__()

class CommandShutdown(Command):
    pass

class CommandWakeWord(Command):
    def __init__(self, wake_word: str, audio_bytes: bytes):
        self.wake_word = wake_word
        self.audio_bytes = audio_bytes

class CommandProcessAudio(Command):
    def __init__(self, audio_bytes: bytes):
        self.audio_bytes = audio_bytes

class CommandProcessText(Command):
    def __init__(self, text: str):
        self.text = text