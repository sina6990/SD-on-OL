# SD-on-OL
In this project, we will investigate the influence of individuals’ geographic location and socio-demographic characteristics on the detection of offensive language in large language models.

## Environment Setup

1. Rename the provided `.env.example` file to `.env`:
	```powershell
	Rename-Item ".env.example" .env
	```
2. Open `.env` and replace `your_huggingface_token` with your actual Hugging Face token:
	```
	HF_TOKEN=your_huggingface_token
	```
3. Save the file. The application will automatically load this token for authentication.

## Installation

To install the required Python packages, run the following command in your terminal from the project directory:

```powershell
pip install -r requirements.txt
```

This will install all dependencies listed in `requirements.txt`.

**Note:** CUDA and PyTorch are not included in `requirements.txt` because they often require manual installation for compatibility with your system and GPU drivers. Please follow the official PyTorch installation instructions for your platform:

https://pytorch.org/get-started/locally/

Make sure to select the correct CUDA version for your hardware and Python environment.
