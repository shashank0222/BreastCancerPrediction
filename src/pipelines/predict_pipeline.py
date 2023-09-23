import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts' , 'preprocessor.pkl')
            model_path = os.path.join('artifacts' , 'model.pkl')

            # loading the pkl files 
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # feature engineering
            data_scaled = preprocessor.transform(features)

            # predicting the new data
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error occured during the prediction of new data")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 meanRadius : float,
                 meanTexture : float,
                 meanPerimeter : float,
                 meanArea : float,
                 meanSmoothness : float, 
                 meanCompactness : float,
                 meanConcavity : float,
                 meanConcavePoints : float, 
                 meanSymmetry : float,
                 meanFractalDimension : float,
                 radiusError : float, 
                 textureError : float, 
                 perimeterError : float, 
                 areaError : float,
                 smoothnessError : float, 
                 compactnessError : float, 
                 concavityError : float,
                 concavePointsError : float, 
                 symmetryError : float, 
                 fractalDimensionError : float,
                 worstRadius : float, 
                 worstTexture : float , 
                 worstPerimeter : float, 
                 worstArea : float,
                 worstSmoothness : float, 
                 worstCompactness : float, 
                 worstConcavity : float,
                 worstConcavePoints : float, 
                 worstSymmetry:float, 
                 worstFractalDimension:float):
        self.meanRadius = meanRadius
        self.meanTexture = meanTexture
        self.meanPerimeter = meanPerimeter
        self.meanArea = meanArea
        self.meanSmoothness = meanSmoothness
        self.meanCompactness = meanCompactness
        self.meanConcavity = meanConcavity
        self.meanConcavePoints = meanConcavePoints
        self.meanSymmetry = meanSymmetry
        self.meanFractalDimension = meanFractalDimension
        self.radiusError = radiusError
        self.textureError = textureError
        self.perimeterError = perimeterError
        self.areaError = areaError
        self.smoothnessError = smoothnessError
        self.compactnessError = compactnessError
        self.concavityError = concavityError
        self.concavePointsError = concavePointsError
        self.symmetryError = symmetryError
        self.fractalDimensionError = fractalDimensionError
        self.worstRadius = worstRadius
        self.worstTexture = worstTexture
        self.worstPerimeter = worstPerimeter
        self.worstArea = worstArea
        self.worstSmoothness = worstSmoothness
        self.worstCompactness = worstCompactness
        self.worstConcavity = worstConcavity
        self.worstConcavePoints = worstConcavePoints
        self.worstSymmetry = worstSymmetry
        self.worstFractalDimension = worstFractalDimension

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                
                'mean radius' : [self.meanRadius],
                'mean texture' : [self.meanTexture],
                'mean perimeter' : [self.meanPerimeter],
                'mean area' : [self.meanArea],
                'mean smoothness' : [self.meanSmoothness],
                'mean compactness' : [self.meanCompactness],
                'mean concavity' : [self.meanConcavity],
                'mean concave points' : [self.meanConcavePoints],
                'mean symmetry' : [self.meanSymmetry],
                'mean fractal dimension' : [self.meanFractalDimension],
                'radius error' : [self.radiusError],
                'texture error' : [self.textureError],
                'perimeter error' : [self.perimeterError],
                'area error' : [self.areaError],
                'smoothness error' : [self.smoothnessError],
                'compactness error' : [self.compactnessError],
                'concavity error' : [self.concavityError],
                'concave points error' : [self.concavePointsError],
                'symmetry error' : [self.symmetryError],
                'fractal dimension error' : [self.fractalDimensionError],
                'worst radius' : [self.worstRadius],
                'worst texture' : [self.worstTexture],
                'worst perimeter' : [self.worstPerimeter],
                'worst area' : [self.worstArea],
                'worst smoothness' : [self.worstSmoothness],
                'worst compactness' : [self.worstCompactness],
                'worst concavity' : [self.worstConcavity],
                'worst concave points' : [self.worstConcavePoints],
                'worst symmetry' : [self.worstSymmetry],
                'worst fractal dimension' : [self.worstFractalDimension]
 
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return df

        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e,sys)





