import streamlit as st
from src.MushroomClassification.pipeline.prediction import Prediction
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load the env variables for the mlflow tracking
load_dotenv()

# Define the input model
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

# Streamlit app
st.title("Mushroom Classification")

with st.form("prediction_form"): 
    bruises = st.selectbox("Bruises", options=['t', 'f'], help="Choose 't' for bruises or 'f' otherwise")
    odor = st.selectbox("Odor", options=['f', 'n', 'o'], help="Choose odor from: 'f' for foul smell, 'n' for no smell, or 'o' for other")
    gill_spacing = st.selectbox("Gill Spacing", options=['w', 'o'], help="Choose gill spacing: 'w' for crowded, 'o' for other")
    gill_size = st.selectbox("Gill Size", options=['b', 'n'], help="Choose gill size: 'b' for broad , 'n' for narrow")
    gill_color = st.selectbox("Gill Color", options=['b', 'o'], help="Choose gill color: 'b' for buff, 'o' for others")
    stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", options=['k', 's', 'o'], help="Choose stalk surface above ring: 'k' for silky, 's' for smooth, 'o' for others")
    stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", options=['k', 's', 'o'], help="Choose stalk surface below ring: 'k' for silky, 's' for smooth, 'o' for others")
    ring_type = st.selectbox("Ring Type", options=['l', 'p', 'o'], help="Choose ring type: 'l' for large, 'p' for pendant, 'o' for others")
    spore_print_color = st.selectbox("Spore Print Color", options=['h', 'k', 'n', 'w', 'o'], help="Choose spore print color: 'h' for chocolate, 'k' for black, 'n' for brown, 'w' for white, 'o' for others")
    population = st.selectbox("Population", options=['v', 'o'], help="Choose population: 'v' for several, 'o' for others")

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_data = Input(
                bruises=bruises,
                odor=odor,
                gill_spacing=gill_spacing,
                gill_size=gill_size,
                gill_color=gill_color,
                stalk_surface_above_ring=stalk_surface_above_ring,
                stalk_surface_below_ring=stalk_surface_below_ring,
                ring_type=ring_type,
                spore_print_color=spore_print_color,
                population=population
            )
            result = Prediction(input_data.dict()).classify()
            if str(result) == '[1]':
                st.error("Prediction Result: Poisonous")
            else :
                st.success("Prediction Result: Edible")
        except ValidationError as e:
            st.error(f"Validation Error: {e}")
