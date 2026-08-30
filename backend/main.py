"""
FastAPI application tying together Agent 6 (Inference) and Agent 7 (Report Generation).
Exposes a single endpoint to upload a medical image and receive a complete medical report.
"""

import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Ensure project root is in sys.path so backend/ config/ etc can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.classifier_service import ClassifierService
from backend.services.report_service import ReportService
from utils.logger import setup_logger, print_step, print_substep
from utils.file_utils import ensure_directory_exists

logger = setup_logger(__name__)

# Constants
UPLOADS_DIR = _PROJECT_ROOT / "outputs" / "uploads"
HEATMAPS_DIR = _PROJECT_ROOT / "outputs" / "heatmaps"
REPORTS_DIR = _PROJECT_ROOT / "outputs" / "reports"
MAX_HISTORY = 5

app = FastAPI(
    title="Medical Vision AI Pipeline",
    description="End-to-End Pipeline for Medical Image Inference and LLM Report Generation.",
    version="1.0.0"
)

# Allow CORS for potential frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Instances
# Loading models is expensive, so we instantiate them once at app startup
classifier_service = None
report_service = None

@app.on_event("startup")
def startup_event():
    """Initialise heavy services on startup."""
    global classifier_service, report_service
    logger.info("Initializing Classifier Service (Agent 6)...")
    classifier_service = ClassifierService()
    logger.info("Initializing Report Service (Agent 7)...")
    report_service = ReportService()
    
    # Ensure directories exist
    ensure_directory_exists(UPLOADS_DIR)
    ensure_directory_exists(HEATMAPS_DIR)
    ensure_directory_exists(REPORTS_DIR)
    logger.info("FastAPI backend started successfully.")


def _cleanup_directory(directory: Path):
    """Keep only the MAX_HISTORY most recent files in the given directory."""
    try:
        if not directory.exists():
            return
            
        # Get all files in the directory sorted by creation/modification time (oldest first)
        files = [f for f in directory.iterdir() if f.is_file()]
        files.sort(key=lambda x: x.stat().st_mtime)
        
        # If we have more than the allowed maximum, delete the oldest ones
        if len(files) > MAX_HISTORY:
            files_to_delete = files[:-MAX_HISTORY]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old file {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")


@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "ok", "message": "Medical API is running."}


@app.post("/api/v1/analyze")
async def analyze_medical_image(file: UploadFile = File(...)):
    """
    Core pipeline endpoint:
    1. Receives and saves the uploaded medical image.
    2. Runs Agent 6 (MedicalClassifierViT) to get predictions and Grad-CAM.
    3. Runs Agent 7 (LLM) to generate the structured report.
    4. Returns the full clinical report payload.
    """
    image_name = file.filename
    logger.info(f"Received API request for image analysis: {image_name}")
    print_step(1, "Receiving Image Data", f"A user has uploaded a medical image: {image_name}", color="\033[1;34m")
    
    # Restrict to only the 5 most recent files extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_path_obj = Path(file.filename)
    if file_path_obj.suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Please upload {allowed_extensions}"
        )
    
    # 2. Cleanup old files before saving the new ones
    _cleanup_directory(UPLOADS_DIR)
    _cleanup_directory(HEATMAPS_DIR)
    _cleanup_directory(REPORTS_DIR)
    
    # 3. Save the uploaded file temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{file_path_obj.stem}_{timestamp}{file_path_obj.suffix}"
    saved_path = UPLOADS_DIR / safe_filename
    
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded image to {saved_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded image to disk.")
        
    # 4. Pipeline Execution
    try:
        # Agent 6: Inference
        logger.info("Starting Agent 6 inference pipeline...")
        prediction_payload = classifier_service.predict(saved_path)
        
        # Agent 7: Report Generation
        logger.info("Starting Agent 7 report generation pipeline...")
        final_report = report_service.generate_report(prediction_payload)
        
        # Save the final report to disk
        report_filename = f"report_{safe_filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')}.json"
        report_path = REPORTS_DIR / report_filename
        import json
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)
        logger.info(f"Saved final medical report to {report_path}")
        
        # 5. Return JSON
        logger.info("Analysis complete. Returning response.")
        return JSONResponse(content=final_report)
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Typically run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
