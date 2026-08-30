import os
import sys

try:
    import gdown
except ImportError:
    print("gdown is not installed. Please run: pip install gdown")
    sys.exit(1)

def main():
    print("📥 MedVision AI: Downloading Pre-trained Vision Transformer Weights...")
    
    # The Google Drive file ID
    file_id = "1xHBmrRjBtu-oyxTtDer4r-vpNkzVA04J"
    
    # Destination directory
    target_dir = os.path.join(os.path.dirname(__file__), "models", "trained")
    os.makedirs(target_dir, exist_ok=True)
    
    # Destination file path
    output_path = os.path.join(target_dir, "medical_classifier.pt")
    
    if os.path.exists(output_path):
        print(f"✅ Weights already exist at: {output_path}")
        print("You are ready to run the application!")
        return

    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Fetching weights from Google Drive to {output_path}")
    print("This is a ~343MB file. Depending on your internet speed, this may take a few minutes...")
    
    # Download using gdown
    gdown.download(url, output_path, quiet=False)
    
    print("🎉 Download complete! You can now start the FastAPI and Streamlit servers.")

if __name__ == "__main__":
    main()
