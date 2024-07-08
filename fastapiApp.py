from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.MushroomClassification.pipeline.prediction import Prediction
from pydantic import BaseModel, constr, Field
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
    bruises: str = Field(..., description="Choose 't' for bruises or 'f' otherwise", pattern='[tf]')
    odor: str = Field(..., description="Choose odor from: 'f' for foul smell, 'n' for no smell, or 'o' for other", pattern='[fno]')
    gill_spacing: str = Field(..., description="Choose gill spacing: 'w' for crowded, 'o' for other", pattern='[wo]')
    gill_size: str = Field(..., description="Choose gill size: 'b' for broad , 'n' for narrow", pattern='[bn]')
    gill_color: str = Field(..., description="Choose gill color: 'b' for buff, 'o' for others", pattern='[bo]')
    stalk_surface_above_ring: str = Field(..., description="Choose stalk surface above ring: 'k' for silky, 's' for smooth, 'o' for others", pattern='[kso]')
    stalk_surface_below_ring: str = Field(..., description="Choose stalk surface below ring: 'k' for silky, 's' for smooth, 'o' for others", pattern='[kso]')
    ring_type: str = Field(..., description="Choose ring type: 'l' for large, 'p' for pendant, 'o' for others", pattern='[lpo]')
    spore_print_color: str = Field(..., description="Choose spore print color: 'h' for chocolate, 'k' for black, 'n' for brown, 'w' for white, 'o' for others", pattern='[hknwo]')
    population: str = Field(..., description="Choose population: 'v' for several, 'o' for others", pattern='[vo]')

    class Config:
        json_schema_extra = {
            "example": {
                "bruises": "f",
                "odor": "f",
                "gill_spacing": "o",
                "gill_size": "b",
                "gill_color": "o",
                "stalk_surface_above_ring": "k",
                "stalk_surface_below_ring": "k",
                "ring_type": "l",
                "spore_print_color": "h",
                "population": "v"
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
    
    uvicorn.run(app, host="0.0.0.0", port=80)