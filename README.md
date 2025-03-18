# NexusVoice

NexusVoice is a simple Python project that uses the `pyaudio` and `openwakeword` libraries.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/madsmith/NexusVoice.git
    cd NexusVoice
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    One of the requirements is ffmpeg for audio processing.  Installation varies by architecture
    but on linux, you may use the following.
    ```sh
    apt install ffmpeg
    ```

    Python dependencies can be installed by running the following.
    
    ```sh
    pip install -r requirements.txt
    ```
    Note: one of the dependencies is pyaudio which on a Mac environment requires portaudio to be installed separately.

    ```sh
    # Install xcode command line tools (if not installed)
    xcode-select --install
    # Using homebrew or package manager of choice
    brew install portaudio
    # Install pyaudio now that requirement is satisfied.
    pip install pyaudio
    ```


## Running the Project

1. Ensure you have the necessary models downloaded:
    ```py
    openwakeword.utils.download_models()
    ```

2. Run the main script:
    ```sh
    python3 NexusVoice.py
    ```

You should see "Hello World!" printed to the console.

## Dependencies

- [pyaudio](http://_vscodecontentref_/1)
- [numpy](http://_vscodecontentref_/2)
- [openwakeword](http://_vscodecontentref_/3)

Make sure to install these dependencies using `pip` as shown in the installation steps.# NexusVoice

NexusVoice is a simple Python project that uses the `pyaudio` and `openwakeword` libraries.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/NexusVoice.git
    cd NexusVoice
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

1. Ensure you have the necessary models downloaded:
    ```py
    openwakeword.utils.download_models()
    ```

2. Run the main script:
    ```sh
    python NexusVoice.py
    ```

You should see "Hello World!" printed to the console.

## Dependencies

- [pyaudio](http://_vscodecontentref_/1)
- [numpy](http://_vscodecontentref_/2)
- [openwakeword](http://_vscodecontentref_/3)

Make sure to install these dependencies using `pip` as shown in the installation steps.