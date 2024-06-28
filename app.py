from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.MushroomClassification.pipeline.prediction import Prediction
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# load the env variables for the mlflow tracking
load_dotenv()

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    bruises: str
    odor: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    ring_type: str
    spore_print_color: str
    population: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "bruises": "f",
                "odor": "f",
                "gill_spacing": "c",
                "gill_size": "b",
                "gill_color": "k",
                "stalk_surface_above_ring": "k",
                "stalk_surface_below_ring": "k",
                "ring_type": "l",
                "spore_print_color": "k",
                "population": "a"
            }
        }


@app.get("/")
async def home():
    return {"message": "Welcome to the Mushroom Classification API --by Sanskar Modi", "/train" : "go to this route to start the training pipeline", "/docs" : "go to this route to be able to send post request on route /predict for classification"}

@app.get("/train")
async def trainRoute():
    # os.system("python main.py")
    os.system("dvc repro")
    return "Training done successfully!"

@app.post("/predict")
async def predict_route(input: Input):
    try:
        # get the result from the Prediction class after passing the input data 
        result = Prediction(input.dict()).classify()
        
        # Return the result
        return JSONResponse({"result": result})
    except Exception as e:
        print(e)
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8080)