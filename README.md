# RedCarpet Lane Detection Visualizer

A tool for visualizing predicted and annotated vanishing points in images from scaled cars

### Requirements

- Python 3.4
- Virtualenv 15.0.1
- OpenCV 3.0 ([Installation guide](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/))

### Setup (For Ubuntu/Linux 64-bit)

1. Create a virtual environment: virtualenv -p python3 env
2. Activate virtual environment: source env/bin/activate
3. Install tensorflow (CPU only mode): pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl
4. Install remaining libraries: pip install -r requirements.txt
5. Change directory: cd env/lib/python3.4/site-packages
6. Create symbolic link for OpenCV 3.0: ln -s /usr/local/lib/python3.4/site-packages/cv2.cpython-34m.so cv2.so

### Configuration

**Data Settings**
- MODELS_PATH (main.py, String): Path to models.
- FREEZED_NAME (main.py, String): Name of model to use.
- READ_PATH (main.py, String): Path to folder with dataset and labels to display.

**Execution Settings**
- AUTOPLAY (main.py, True/False): Run sequence of images automatically.

### Running

1. Execute main file: python main.py